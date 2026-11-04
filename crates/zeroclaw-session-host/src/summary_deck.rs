//! Live runtime policy and provider composition for the kernel summary deck.
use super::*;
use zeroclaw_api::session_events::capture::ContextSummaryExecutor;
use zeroclaw_kernel::session::summary_deck::{ResolvedSummaryDeck, SummaryDeckRuntime};

pub(super) struct Runtime {
    pub config: LiveConfigHandle,
    pub agent: String,
    pub session: SessionId,
}

impl SummaryDeckRuntime for Runtime {
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
        let provider_ref = if profile.summary_deck.provider.trim().is_empty() {
            active_ref.to_owned()
        } else {
            profile.summary_deck.provider.trim().to_owned()
        };
        let model = if profile.summary_deck.provider.trim().is_empty() {
            Some(active_model)
        } else {
            None
        };
        let source_window = config
            .resolved_context_limits_for_route(
                &self.agent,
                &provider_ref,
                &zeroclaw_providers::factory::resolve_session_model_name(
                    &config,
                    &provider_ref,
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
        let provider_ref = if profile.summary_deck.provider.trim().is_empty() {
            selected
                .map(|(provider, _)| provider.to_owned())
                .unwrap_or_else(|| format!("{family}.{alias}"))
        } else {
            profile.summary_deck.provider.trim().to_owned()
        };
        let model = if profile.summary_deck.provider.trim().is_empty() {
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
                let messages: Vec<_> = conversation
                    .iter()
                    .flat_map(zeroclaw_api::model_provider::ConversationMessage::chat_messages)
                    .collect();
                let tracker = super::summary_cost::admit(
                    &live,
                    &agent,
                    &provider_ref,
                    &binding.model,
                    &messages,
                )?;
                let accounting = zeroclaw_providers::dispatch::AccountedChatScope::new();
                let response = accounting
                    .scope(
                        zeroclaw_providers::dispatch::ProviderDispatch::from_ref(
                            binding.provider.as_ref(),
                        )
                        .chat(
                            zeroclaw_api::model_provider::ChatRequest {
                                messages: &messages,
                                tools: None,
                                thinking: None,
                            },
                            &binding.model,
                            Some(0.0),
                        ),
                    )
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
                let response = response?;
                anyhow::ensure!(
                    response.tool_calls.is_empty(),
                    "summary_deck_unexpected_tool_calls"
                );
                Ok(response.text_or_empty().to_owned())
            })
        }))
    }
}
