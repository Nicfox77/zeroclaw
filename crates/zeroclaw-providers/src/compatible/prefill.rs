//! Bounded cache-prefill inference through the ordinary chat request builder.
use super::*;
use anyhow::{Context, ensure};
use std::time::{Duration, Instant};

pub(crate) fn local_endpoint(endpoint: &str) -> anyhow::Result<()> {
    let url = reqwest::Url::parse(endpoint)?;
    ensure!(
        matches!(url.scheme(), "http" | "https")
            && matches!(url.host_str(), Some("127.0.0.1" | "[::1]" | "localhost"))
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "cache_warm_requires_loopback_endpoint"
    );
    Ok(())
}

impl OpenAiCompatibleModelProvider {
    pub(super) async fn prefill_llamacpp(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
        inactivity_secs: u64,
    ) -> anyhow::Result<()> {
        ensure!(
            self.name == "llama.cpp",
            "provider_prompt_prefill_unsupported"
        );
        let url = self.chat_completions_url();
        local_endpoint(&url)?;
        let inactivity = Duration::from_secs(inactivity_secs);
        ensure!(!inactivity.is_zero(), "cache_warm_invalid_inactivity");
        let credential = self.resolve_credential().await?;
        // Warming has no independent filesystem/media capability. Stop at
        // the start of the first image-bearing turn; normal foreground image
        // preparation remains responsible for path authorization and decoding.
        let end = request
            .messages
            .iter()
            .position(|message| {
                message
                    .attachments
                    .iter()
                    .any(|a| a.kind == AttachmentKind::Image)
            })
            .map(|image| {
                request.messages[..=image]
                    .iter()
                    .rposition(|m| m.role == "user")
                    .unwrap_or(0)
            })
            .unwrap_or(request.messages.len());
        if end == 0 {
            return Ok(());
        }
        let normalized = self
            .normalize_messages_for_upstream(&request.messages[..end])
            .await?;
        let merge = self.effective_merge_system(model);
        let (messages, merged) = Self::flatten_system_messages(&normalized, merge);
        let messages = if self.native_tool_calling {
            messages
        } else {
            self.strip_native_tool_messages(&messages)
        };
        let tools = self.convert_tool_specs_for_model(request.tools, model);
        let mut payload = serde_json::to_value(self.build_native_tool_chat_request(
            &messages,
            tools,
            model,
            temperature,
            !merge,
            merged,
            request.thinking,
        ))?;
        // This server samples one token even for n_predict=0. Use an explicit
        // one-token ceiling, discard it, and never publish an assistant message.
        // n_predict overrides max_tokens; force both after profile extras.
        payload["max_tokens"] = serde_json::json!(1);
        payload["n_predict"] = serde_json::json!(1);
        payload["stream"] = serde_json::json!(true);
        payload["stream_options"] = serde_json::json!({"include_usage":true});
        payload["cache_prompt"] = serde_json::json!(true);
        payload["return_progress"] = serde_json::json!(true);
        payload["id_slot"] = serde_json::json!(0);
        // This operation is local-only, bypasses proxies and never follows a
        // redirect with private conversation content. Reuse current credentials,
        // headers, TLS material and the normal message/tool conversion above.
        let client = self
            .add_tls_cert_to_builder(
                Client::builder()
                    .no_proxy()
                    .redirect(reqwest::redirect::Policy::none())
                    .connect_timeout(Duration::from_secs(10))
                    .read_timeout(inactivity),
            )
            .build()?;
        let mut send = client.post(url).json(&payload);
        for (name, value) in &self.extra_headers {
            send = send.header(name, value);
        }
        if let Some(ua) = &self.user_agent {
            send = send.header(USER_AGENT, ua);
        }
        let send = self.apply_auth_header(send, credential.as_deref())?;
        let started = Instant::now();
        ::zeroclaw_log::record!(
            INFO,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Start)
                .with_attrs(serde_json::json!({"model":model})),
            "llamacpp_cache_warm_started"
        );
        let response = tokio::time::timeout(inactivity, send.send())
            .await
            .context("cache_warm_inactive")??;
        ensure!(
            response.status().is_success(),
            "cache_warm_http_status_{}",
            response.status().as_u16()
        );
        let mut stream = response.bytes_stream();
        let mut deadline = tokio::time::Instant::now() + inactivity;
        let mut line = Vec::new();
        let mut processed = 0;
        let mut usage = serde_json::Value::Null;
        let mut output_bytes = 0;
        let mut finished = false;
        let mut done = false;
        while let Some(chunk) = tokio::time::timeout_at(deadline, stream.next())
            .await
            .context("cache_warm_inactive")?
        {
            for byte in chunk? {
                if byte != b'\n' {
                    ensure!(line.len() < 65536, "cache_warm_frame_too_large");
                    line.push(byte);
                    continue;
                }
                let text = std::str::from_utf8(&line)?.trim();
                if let Some(data) = text.strip_prefix("data:").map(str::trim) {
                    if data == "[DONE]" {
                        done = true;
                    } else if !data.is_empty() {
                        let event: serde_json::Value = serde_json::from_str(data)?;
                        ensure!(event.get("error").is_none(), "cache_warm_protocol_error");
                        if let Some(n) = event
                            .pointer("/prompt_progress/processed")
                            .and_then(serde_json::Value::as_u64)
                            && n > processed
                        {
                            processed = n;
                            deadline = tokio::time::Instant::now() + inactivity;
                        }
                        if let Some(choices) = event["choices"].as_array() {
                            for choice in choices {
                                for field in [
                                    "content",
                                    "reasoning_content",
                                    "reasoning",
                                    "tool_calls",
                                    "function_call",
                                ] {
                                    let value = &choice["delta"][field];
                                    if !value.is_null() && value.as_str() != Some("") {
                                        output_bytes += serde_json::to_vec(value)?.len();
                                    }
                                }
                                ensure!(output_bytes <= 8192, "cache_warm_output_limit");
                                finished |= choice["finish_reason"].is_string();
                            }
                        }
                        if event["usage"].is_object() {
                            ensure!(
                                event["usage"]["completion_tokens"]
                                    .as_u64()
                                    .is_some_and(|n| n <= 1),
                                "cache_warm_generated_output"
                            );
                            usage = event["usage"].clone();
                        }
                    }
                }
                line.clear();
            }
            if done {
                break;
            }
        }
        ensure!(
            done && finished && usage["completion_tokens"].as_u64().is_some_and(|n| n <= 1),
            "cache_warm_incomplete_stream"
        );
        ::zeroclaw_log::record!(INFO, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Complete)
            .with_attrs(serde_json::json!({"model":model,"duration_ms":started.elapsed().as_millis(),
                "prompt_tokens":usage["prompt_tokens"],"cached_tokens":usage.pointer("/prompt_tokens_details/cached_tokens"),
                "processed_tokens":processed,"generated_tokens":usage["completion_tokens"]})), "llamacpp_cache_warm_completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, body::Body, response::Response, routing::post};
    use std::sync::{Arc, Mutex};
    use zeroclaw_api::tool::ToolSpec;

    fn terminal() -> String {
        "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n".into()
    }
    fn provider(url: &str) -> OpenAiCompatibleModelProvider {
        OpenAiCompatibleModelProvider::builder("fixture")
            .display_name("llama.cpp")
            .base_url(url)
            .auth_style(AuthStyle::Bearer)
            .credential(Some("fixture"))
            .with_native_tools(true)
            .max_tokens(Some(16))
            .build()
    }
    type Requests = Arc<Mutex<Vec<serde_json::Value>>>;
    async fn server(frames: Vec<(u64, String)>) -> (String, Requests, tokio::task::JoinHandle<()>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let capture = seen.clone();
        let app = Router::new().route("/chat/completions", post(move |Json(body): Json<serde_json::Value>| {
            let capture = capture.clone(); let frames = frames.clone();
            async move {
                capture.lock().unwrap().push(body.clone());
                if body["return_progress"] == true {
                    let chunks = futures_util::stream::unfold(frames.into_iter(), |mut it| async {
                        let (delay, frame) = it.next()?;
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        Some((Ok::<_, std::io::Error>(frame), it))
                    });
                    Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(chunks)).unwrap()
                } else {
                    Response::builder().header("content-type", "application/json")
                        .body(Body::from(r#"{"choices":[{"message":{"role":"assistant","content":"ordinary answer"},"finish_reason":"stop"}]}"#)).unwrap()
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = ::zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), seen, task)
    }

    #[tokio::test]
    async fn prefill_preserves_real_chat_messages_and_tools_but_never_runs_a_tool() {
        let (url, seen, server) = server(vec![(0, terminal())]).await;
        let p = provider(&url);
        let messages = vec![
            ChatMessage::system("Stable system prefix"),
            ChatMessage::user("Current history"),
        ];
        let tools = vec![ToolSpec::new(
            "fixture_lookup",
            "Read fixture",
            serde_json::json!({"type":"object"}),
        )];
        let request = ProviderChatRequest {
            messages: &messages,
            tools: Some(&tools),
            thinking: None,
        };
        p.chat(request, "fixture", Some(0.3)).await.unwrap();
        p.prefill_prompt(request, "fixture", Some(0.3), 2)
            .await
            .unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0]["messages"], seen[1]["messages"]);
        assert_eq!(seen[0]["tools"], seen[1]["tools"]);
        assert_eq!(seen[1]["max_tokens"], 1);
        assert_eq!(seen[1]["n_predict"], 1);
        assert_eq!(seen[1]["cache_prompt"], true);
        assert_eq!(seen[1]["id_slot"], 0);
        server.abort();
    }

    #[tokio::test]
    async fn actual_prefill_progress_extends_inactivity_but_keepalives_do_not() {
        for real_progress in [true, false] {
            let mut frames = Vec::new();
            for n in 1..=4 {
                frames.push((
                    400,
                    if real_progress {
                        format!("data: {{\"prompt_progress\":{{\"processed\":{n}}}}}\n\n")
                    } else {
                        ": keepalive\n\n".into()
                    },
                ));
            }
            frames.push((0, terminal()));
            let (url, _, server) = server(frames).await;
            let p = provider(&url);
            let messages = vec![ChatMessage::user("fixture")];
            let result = p
                .prefill_prompt(
                    ProviderChatRequest {
                        messages: &messages,
                        tools: None,
                        thinking: None,
                    },
                    "fixture",
                    None,
                    1,
                )
                .await;
            assert_eq!(result.is_ok(), real_progress, "{result:?}");
            server.abort();
        }
    }

    #[tokio::test]
    async fn many_prompt_progress_frames_do_not_consume_the_generated_output_allowance() {
        let mut frames = String::new();
        for n in 1..=1000 {
            frames.push_str(&format!("data: {{\"choices\":[{{\"delta\":{{\"role\":\"assistant\",\"content\":null}}}}],\"prompt_progress\":{{\"processed\":{n}}}}}\n\n"));
        }
        frames.push_str(&terminal());
        let (url, _, server) = server(vec![(0, frames)]).await;
        let p = provider(&url);
        let messages = vec![ChatMessage::user("fixture")];
        p.prefill_prompt(
            ProviderChatRequest {
                messages: &messages,
                tools: None,
                thinking: None,
            },
            "fixture",
            None,
            2,
        )
        .await
        .unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn warming_stops_before_media_instead_of_loading_unapproved_paths() {
        let (url, seen, server) = server(vec![(0, terminal())]).await;
        let p = provider(&url);
        let messages = vec![
            ChatMessage::system("Stable prefix"),
            ChatMessage::user("Earlier text"),
            ChatMessage::assistant("Earlier answer"),
            ChatMessage::user("New image").with_attachments(vec![AttachmentReference {
                kind: AttachmentKind::Image,
                target: "/unapproved/nonexistent/image.png".into(),
            }]),
        ];
        p.prefill_prompt(
            ProviderChatRequest {
                messages: &messages,
                tools: None,
                thinking: None,
            },
            "fixture",
            None,
            2,
        )
        .await
        .unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0]["messages"].as_array().unwrap().len(), 3);
        assert!(!seen[0].to_string().contains("unapproved"));
        server.abort();
    }

    #[tokio::test]
    async fn redirects_never_forward_the_private_warm_prompt() {
        let forwarded = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = forwarded.clone();
        let target = axum::Router::new().route(
            "/stolen",
            axum::routing::post(move || {
                let count = count.clone();
                async move {
                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    "unexpected"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let second = ::zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, target).await.unwrap();
        });
        let app = Router::new().route(
            "/chat/completions",
            post(move || async move {
                Response::builder()
                    .status(307)
                    .header("location", format!("http://{address}/stolen"))
                    .body(Body::empty())
                    .unwrap()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let first = ::zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let p = provider(&format!("http://{address}"));
        let messages = vec![ChatMessage::user("private fixture")];
        let error = p
            .prefill_prompt(
                ProviderChatRequest {
                    messages: &messages,
                    tools: None,
                    thinking: None,
                },
                "fixture",
                None,
                2,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("307"));
        assert_eq!(forwarded.load(std::sync::atomic::Ordering::SeqCst), 0);
        first.abort();
        second.abort();
    }

    #[tokio::test]
    async fn failed_incomplete_oversized_or_unbounded_generation_is_not_success() {
        for body in [
            "data: [DONE]\n\n".to_string(),
            "data: {}\n\n".into(),
            terminal().replace("\"completion_tokens\":1", "\"completion_tokens\":2"),
            format!("data: {}\n\n", "x".repeat(65537)),
        ] {
            let (url, _, server) = server(vec![(0, body)]).await;
            let p = provider(&url);
            let messages = vec![ChatMessage::user("fixture")];
            assert!(
                p.prefill_prompt(
                    ProviderChatRequest {
                        messages: &messages,
                        tools: None,
                        thinking: None
                    },
                    "fixture",
                    None,
                    1
                )
                .await
                .is_err()
            );
            server.abort();
        }
        for endpoint in [
            "http://example.org/v1",
            "http://user@127.0.0.1/v1",
            "http://127.0.0.1/v1?key=fixture",
        ] {
            assert!(local_endpoint(endpoint).is_err());
        }
    }

    /// Run against an isolated CPU-only server, never the user's active slot.
    #[tokio::test]
    #[ignore = "requires an isolated llama.cpp test server via ZEROCLAW_PREFILL_TEST_URL"]
    async fn real_llamacpp_retains_idle_cache_and_reuses_it_for_an_appended_turn() {
        let endpoint =
            std::env::var("ZEROCLAW_PREFILL_TEST_URL").expect("isolated fixture endpoint");
        let p = provider(&endpoint);
        let mut messages = vec![
            ChatMessage::system("Stable system and tool prompt."),
            ChatMessage::user("A story about a small test. ".repeat(12)),
            ChatMessage::assistant("Ready."),
        ];
        p.prefill_prompt(
            ProviderChatRequest {
                messages: &messages,
                tools: None,
                thinking: None,
            },
            "prefill-fixture",
            None,
            30,
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
        messages.push(ChatMessage::user("Continue."));
        let response = p
            .chat(
                ProviderChatRequest {
                    messages: &messages,
                    tools: None,
                    thinking: None,
                },
                "prefill-fixture",
                None,
            )
            .await
            .unwrap();
        let usage = response.usage.expect("real usage");
        assert!(usage.input_tokens.unwrap_or(0) > 100);
        assert!(
            usage.cached_input_tokens.unwrap_or(0) > 100,
            "cache must survive idleness and the appended user turn: {usage:?}"
        );
    }
}
