//! Live runtime policy and provider composition for the kernel summary deck.
use super::*;
use futures_util::{StreamExt, stream::BoxStream};
use zeroclaw_api::model_provider::{
    ChatRequest, ProviderOutputBlock, StreamEvent, StreamOptions, StreamResult,
};
use zeroclaw_api::session_events::capture::ContextSummaryExecutor;
use zeroclaw_kernel::session::summary_deck::{ResolvedSummaryDeck, SummaryDeckRuntime};

pub(super) struct Runtime {
    pub config: LiveConfigHandle,
    pub agent: String,
    pub session: SessionId,
}

impl SummaryDeckRuntime for Runtime {
    fn main_session(&self) -> anyhow::Result<Option<SessionId>> {
        let config = self.config.read();
        let Some(agent) = config.agent(&self.agent) else {
            return Ok(None);
        };
        if !agent.main_session.enabled {
            return Ok(None);
        }
        Ok(Some(SessionId::parse(
            zeroclaw_config::main_session::session_id(&self.agent),
        )?))
    }
    fn resolve(&self) -> anyhow::Result<Option<ResolvedSummaryDeck>> {
        self.policy(None)
    }
    fn resolve_for_route(
        &self,
        provider: &str,
        model: &str,
    ) -> anyhow::Result<Option<ResolvedSummaryDeck>> {
        self.policy(Some((provider, model)))
    }
    fn executor(&self) -> anyhow::Result<ContextSummaryExecutor<'static>> {
        self.build_executor(None)
    }
    fn executor_for_route(
        &self,
        provider: &str,
        model: &str,
    ) -> anyhow::Result<ContextSummaryExecutor<'static>> {
        self.build_executor(Some((provider, model)))
    }
}

impl Runtime {
    // The deck-specific override wins; otherwise share the canonical agent /
    // compression summary route. Resolve from live config at the use boundary.
    fn configured_provider(&self, config: &zeroclaw_config::schema::Config) -> Option<String> {
        config
            .runtime_profile_for_agent(&self.agent)
            .map(|profile| profile.summary_deck.provider.trim())
            .filter(|provider| !provider.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                config
                    .effective_summary_provider(&self.agent)
                    .map(|provider| provider.trim().to_owned())
            })
    }

    fn policy(
        &self,
        selected: Option<(&str, &str)>,
    ) -> anyhow::Result<Option<ResolvedSummaryDeck>> {
        let config = self.config.read();
        let Some(profile) = config.runtime_profile_for_agent(&self.agent) else {
            return Ok(None);
        };
        if !profile.summary_deck.enabled {
            return Ok(None);
        }
        let agent = config
            .resolved_agent_config(&self.agent)
            .ok_or_else(|| anyhow::Error::msg("summary_deck_agent_unavailable"))?;
        anyhow::ensure!(agent.enabled, "summary_deck_agent_disabled");
        let (family, alias, entry) = config
            .resolved_model_provider_for_agent(&self.agent)
            .ok_or_else(|| anyhow::Error::msg("summary_deck_provider_unavailable"))?;
        let default_ref = format!("{family}.{alias}");
        let (active_ref, active_model) =
            selected.unwrap_or((&default_ref, entry.model.as_deref().unwrap_or("")));
        let budget = config
            .resolved_context_limits_for_route(&self.agent, active_ref, active_model)
            .context_token_budget;
        let configured = self.configured_provider(&config);
        let provider_ref = configured.as_deref().unwrap_or(active_ref);
        let model = if configured.is_none() {
            Some(active_model)
        } else {
            None
        };
        let source_window = config
            .resolved_context_limits_for_route(
                &self.agent,
                provider_ref,
                &zeroclaw_providers::factory::resolve_session_model_name(
                    &config,
                    provider_ref,
                    model,
                )?,
            )
            .model_context_window;
        Ok(Some(ResolvedSummaryDeck {
            config: profile.summary_deck.clone(),
            // Leave a margin for tool schemas, recalled context, images and
            // estimator uncertainty. The ordinary dispatch capacity guard stays authoritative.
            context_budget: budget.saturating_mul(85) / 100,
            max_turns: agent.resolved.max_history_messages,
            source_budget: source_window.saturating_mul(60) / 100,
        }))
    }

    fn build_executor(
        &self,
        selected: Option<(&str, &str)>,
    ) -> anyhow::Result<ContextSummaryExecutor<'static>> {
        let config = self.config.read();
        let profile = config
            .runtime_profile_for_agent(&self.agent)
            .ok_or_else(|| anyhow::Error::msg("summary_deck_profile_unavailable"))?;
        anyhow::ensure!(profile.summary_deck.enabled, "summary_deck_disabled");
        let (family, alias, _) = config
            .resolved_model_provider_for_agent(&self.agent)
            .ok_or_else(|| anyhow::Error::msg("summary_deck_provider_unavailable"))?;
        let configured = self.configured_provider(&config);
        let provider_ref = if let Some(provider) = configured.as_ref() {
            provider.clone()
        } else {
            selected
                .map(|(provider, _)| provider.to_owned())
                .unwrap_or_else(|| format!("{family}.{alias}"))
        };
        let model = if configured.is_none() {
            selected.map(|(_, model)| model)
        } else {
            None
        };
        let binding = zeroclaw_providers::factory::build_session_model_provider(
            &config,
            &provider_ref,
            model,
        )?;
        let live = self.config.clone();
        let agent = self.agent.clone();
        let session = self.session.to_string();
        Ok(Box::new(move |conversation| {
            Box::pin(async move {
                let mut messages: Vec<_> = conversation
                    .iter()
                    .flat_map(zeroclaw_api::model_provider::ConversationMessage::chat_messages)
                    .collect();
                let workspace = live.read().agent_workspace_dir(&agent);
                let orientation = zeroclaw_infra::memory_orientation::load(&workspace)?;
                if !orientation.is_empty() {
                    messages.insert(
                        0,
                        zeroclaw_api::model_provider::ChatMessage::user(orientation),
                    );
                }
                let tracker = super::summary_cost::admit(
                    &live,
                    &agent,
                    &provider_ref,
                    &binding.model,
                    &messages,
                )?;
                let accounting = zeroclaw_providers::dispatch::AccountedChatScope::new();
                let response = accounting
                    .scope(consume_summary(
                        zeroclaw_providers::dispatch::ProviderDispatch::from_ref(
                            binding.provider.as_ref(),
                        )
                        .stream_chat(
                            ChatRequest {
                                messages: &messages,
                                tools: None,
                                thinking: None,
                            },
                            &binding.model,
                            Some(0.0),
                            StreamOptions::new(true).with_token_count(),
                        ),
                    ))
                    .await;
                if response.is_ok() {
                    accounting.mark_logical_success();
                }
                super::summary_cost::record(
                    &live,
                    tracker.as_ref(),
                    &agent,
                    &session,
                    accounting.take(),
                )?;
                response
            })
        }))
    }
}

async fn consume_summary(
    mut stream: BoxStream<'static, StreamResult<StreamEvent>>,
) -> anyhow::Result<String> {
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        let event = event?;
        if event.has_generation_progress() {
            zeroclaw_kernel::turn_activity::Activity::progress_current();
        }
        match event {
            StreamEvent::TextDelta(chunk) => text.push_str(&chunk.delta),
            StreamEvent::OutputBlock(ProviderOutputBlock::Text { text: block }) => {
                text.push_str(&block)
            }
            StreamEvent::OutputBlock(
                ProviderOutputBlock::ProtocolMetadata { .. }
                | ProviderOutputBlock::Reasoning { .. },
            ) => {}
            StreamEvent::Final => return Ok(text),
            StreamEvent::ToolCall(_)
            | StreamEvent::PreExecutedToolCall { .. }
            | StreamEvent::PreExecutedToolResult { .. } => {
                anyhow::bail!("summary_deck_unexpected_tool_calls")
            }
            StreamEvent::OutputBlock(_) => anyhow::bail!("summary_deck_unexpected_output_block"),
            _ => {}
        }
    }
    anyhow::bail!("summary_deck_incomplete_stream")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use std::time::Duration;
    use zeroclaw_api::model_provider::StreamChunk;

    #[tokio::test(start_paused = true)]
    async fn summary_stream_counts_real_progress_but_not_keepalives() {
        for progressing in [true, false] {
            let stream = stream::unfold(0, move |n| async move {
                tokio::time::sleep(Duration::from_millis(500)).await;
                let event = if !progressing {
                    StreamEvent::ProtocolMetadata(serde_json::json!({"kind":"keepalive"}))
                } else if n < 8 {
                    StreamEvent::ThinkingDelta("private reasoning".into())
                } else if n == 8 {
                    StreamEvent::TextDelta(StreamChunk::delta("I checked the result."))
                } else {
                    StreamEvent::Final
                };
                Some((Ok(event), n + 1))
            })
            .boxed();
            let cancel = tokio_util::sync::CancellationToken::new();
            let started = tokio::time::Instant::now();
            let result =
                zeroclaw_kernel::turn_activity::run(Some(1), &cancel, consume_summary(stream))
                    .await;
            if progressing {
                assert_eq!(result.unwrap(), "I checked the result.");
                assert!(started.elapsed() > Duration::from_secs(1));
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .is::<zeroclaw_kernel::turn_activity::IdleTimeout>()
                );
                assert!(cancel.is_cancelled());
            }
        }
    }

    #[tokio::test]
    async fn summary_stream_requires_final_and_propagates_provider_errors() {
        let error = consume_summary(stream::empty().boxed()).await.unwrap_err();
        assert_eq!(error.to_string(), "summary_deck_incomplete_stream");
        let error = consume_summary(
            stream::iter([Err(zeroclaw_api::model_provider::StreamError::Http(
                "fixture".into(),
            ))])
            .boxed(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("fixture"));
    }

    #[test]
    fn summary_route_reads_current_overrides_in_precedence_order() {
        let mut config = config_fixture();
        let live = zeroclaw_config::live::LiveConfig::new(config.clone());
        let runtime = Runtime {
            config: live.handle(),
            agent: "fixture".into(),
            session: SessionId::parse("fixture-session").unwrap(),
        };
        assert_eq!(runtime.configured_provider(&config), None);
        config
            .runtime_profiles
            .get_mut("fixture")
            .unwrap()
            .context_compression
            .summary_provider = "custom.profile".into();
        assert_eq!(
            runtime.configured_provider(&config).as_deref(),
            Some("custom.profile")
        );
        config.agents.get_mut("fixture").unwrap().summary_provider = "custom.agent".into();
        assert_eq!(
            runtime.configured_provider(&config).as_deref(),
            Some("custom.agent")
        );
        config
            .runtime_profiles
            .get_mut("fixture")
            .unwrap()
            .summary_deck
            .provider = "custom.deck".into();
        assert_eq!(
            runtime.configured_provider(&config).as_deref(),
            Some("custom.deck")
        );
    }

    fn config_fixture() -> zeroclaw_config::schema::Config {
        let mut config = zeroclaw_config::schema::Config::default();
        config.agents.insert(
            "fixture".into(),
            zeroclaw_config::schema::AliasedAgentConfig {
                runtime_profile: "fixture".into(),
                ..Default::default()
            },
        );
        config
            .runtime_profiles
            .insert("fixture".into(), Default::default());
        config
    }
}
