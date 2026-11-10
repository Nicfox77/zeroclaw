use super::*;
use zeroclaw_api::model_provider::{ProviderContentPosition, ToolRoundPolicy};

async fn crash_with_text(
    port: &EngineSession,
    protocol: bool,
    purpose: ProviderPurpose,
    text: bool,
) {
    let write = port.clone();
    let ready = Arc::new(tokio::sync::Notify::new());
    let signal = ready.clone();
    let mut execution = Box::pin(port.execute(
        input(),
        Box::pin(async move {
            let mut user = ChatMessage::user("Keep this request.");
            user.recalled_memory = Some(zeroclaw_api::model_provider::RecalledMemoryContext {
                entries: vec![zeroclaw_api::model_provider::RecalledMemoryEntry {
                    key: "fixture-memory".into(),
                    content: "An existing historical recall snapshot.".into(),
                    stored_on: None,
                }],
            });
            write
                .replace_context(
                    ContextSnapshot {
                        messages: vec![user.into_conversation()],
                        history_has_trim_breadcrumb: Some(true),
                    },
                    "fixture_input",
                )
                .await?;
            let admission = current_provider_admission().expect("scope");
            let admission = if purpose == ProviderPurpose::Main {
                admission
            } else {
                admission
                    .begin_context_transform(ContextTransformIntent {
                        reason_key: "fixture_summary".into(),
                        algorithm_version: "fixture/v1".into(),
                        budget: 100,
                    })
                    .await?
                    .expect("transform")
                    .provider_capture(purpose)?
            };
            let observer = admission
                .admit(ProviderDispatchIntent {
                    requested_route: None,
                    previous_attempt_event_id: None,
                    provider_ref: "fixture".into(),
                    model: "fixture".into(),
                    messages: vec![ChatMessage::user("Keep this request.")],
                    tools: None,
                    temperature: None,
                    thinking: None,
                    tool_round_policy: ToolRoundPolicy::Batch,
                })
                .await?;
            let mut payloads = vec![];
            let mut operations = vec![];
            let values = [
                "A secret reasoning trace",
                "First visible sentence. ",
                "Second unfinished",
            ];
            // Protocol coordinates must restore order even when independent
            // output items arrive out of order.
            for index in if protocol { [0, 2, 1] } else { [0, 1, 2] } {
                let value = values[index];
                if !text && index > 0 {
                    continue;
                }
                let leaf = PayloadRecord::new(
                    PayloadId::new(),
                    if index == 0 {
                        "session.reasoning/v1"
                    } else {
                        "session.text/v1"
                    }
                    .into(),
                    value.as_bytes().to_vec(),
                )?;
                let block = if index == 0 {
                    ResponseBlock::Reasoning {
                        representation_kind: "thinking".into(),
                        payload_ref: StoredPayloadRef::from(&leaf),
                        completion: ContentCompletion::Partial,
                        replay_requirement: "display_only".into(),
                    }
                } else {
                    ResponseBlock::Text {
                        text_ref: StoredPayloadRef::from(&leaf),
                    }
                };
                let descriptor = PayloadRecord::json("session.evidence/v1", &block)?;
                if protocol {
                    operations.push(ProviderObservation::ProtocolSegmentObserved {
                        position: ProviderContentPosition {
                            output_index: index as u64,
                            content_index: Some(0),
                        },
                        component: block.component().0.into(),
                        byte_offset: 0,
                        segment_ref: StoredPayloadRef::from(&descriptor),
                    });
                } else {
                    operations.extend([
                        ProviderObservation::BlockOpened {
                            block_index: index,
                            descriptor_ref: StoredPayloadRef::from(&descriptor),
                        },
                        ProviderObservation::BytesAppended {
                            block_index: index,
                            component: block.component().0.into(),
                            byte_offset: 0,
                            bytes_ref: StoredPayloadRef::from(&leaf),
                        },
                        ProviderObservation::BlockFinalized {
                            block_index: index,
                            finalized_block_ref: StoredPayloadRef::from(&descriptor),
                        },
                    ]);
                }
                payloads.extend([leaf, descriptor]);
            }
            let observation = PayloadRecord::json(
                "session.provider-observation/v1",
                &ProviderObservationPayload { operations },
            )?;
            let observation_ref = StoredPayloadRef::from(&observation);
            payloads.push(observation);
            observer
                .observe(
                    EventData::ProviderResponseProgressObserved { observation_ref },
                    payloads,
                )
                .await?;
            signal.notify_one();
            std::future::pending().await
        }),
    ));
    tokio::select! {
        () = ready.notified() => {},
        result = &mut execution => panic!("fixture failed before crash: {result:?}"),
    }
    // Drop the task rather than cooperatively cancelling it: no terminal can
    // be written, just as when the owning process is killed.
    drop(execution);
}

#[tokio::test]
async fn crash_recovery_preserves_visible_partial_once_across_reopen_and_next_turn() {
    for backend in ["sqlite", "jsonl"] {
        for protocol in [false, true] {
            let dir = tempfile::TempDir::new().expect("directory");
            let (port, authority) = session(dir.path(), backend);
            crash_with_text(&port, protocol, ProviderPurpose::Main, true).await;
            let stream = port.stream.clone();
            let branch = port.branch.clone();
            drop(port);
            let store = zeroclaw_infra::session_events::open_event_store(dir.path(), backend)
                .expect("restart");
            let owner = Arc::new(SessionCoordinator::new(store));
            owner
                .recover_interrupted_stream(
                    stream.clone(),
                    "crash".into(),
                    Arc::new(ElicitationOwnerLost),
                )
                .await
                .expect("recovery");
            let head = owner.head(&stream).expect("head");
            owner
                .recover_interrupted_stream(
                    stream.clone(),
                    "another-recovery-command".into(),
                    Arc::new(ElicitationOwnerLost),
                )
                .await
                .expect("repeat recovery");
            assert_eq!(head, owner.head(&stream).expect("unchanged head"));
            let port = EngineSession::new(
                owner,
                stream,
                branch,
                PrincipalId::shared_operator(),
                authority,
            );
            let restored = port
                .restore_context()
                .await
                .expect("restore")
                .expect("snapshot");
            assert_eq!(restored.history_has_trim_breadcrumb, Some(true));
            let chats = restored.chat_projection();
            assert_eq!(chats.len(), 2);
            assert_eq!(chats[0].content, "Keep this request.");
            assert_eq!(
                chats[0]
                    .recalled_memory
                    .as_ref()
                    .expect("inherited recall")
                    .entries[0]
                    .key,
                "fixture-memory"
            );
            assert_eq!(chats[1].role, "assistant");
            assert!(
                chats[1]
                    .content
                    .starts_with("First visible sentence. Second unfinished\n\n[Interrupted:")
            );
            assert!(!chats[1].content.contains("secret reasoning"));
            assert!(chats[1].reasoning_content.is_none());
            let expected = canonical_json(&restored).expect("snapshot");
            let next = port.clone();
            port.execute(
                input(),
                Box::pin(async move {
                    let context = next.restore_context().await?.expect("next context");
                    assert_eq!(canonical_json(&context)?, expected);
                    next.replace_context(context, "next_turn_history").await?;
                    Ok("continued".into())
                }),
            )
            .await
            .expect("next turn sees partial");
            let context = port
                .restore_context()
                .await
                .expect("restore again")
                .expect("snapshot");
            assert_eq!(
                context
                    .chat_projection()
                    .iter()
                    .filter(|m| m.content.contains("[Interrupted:"))
                    .count(),
                1
            );
        }
    }
}

#[tokio::test]
async fn crash_recovery_does_not_promote_reasoning_or_summary_output() {
    for backend in ["sqlite", "jsonl"] {
        for (purpose, text) in [
            (ProviderPurpose::Main, false),
            (ProviderPurpose::Summary, true),
        ] {
            let dir = tempfile::TempDir::new().expect("directory");
            let (port, _) = session(dir.path(), backend);
            crash_with_text(&port, true, purpose, text).await;
            port.owner
                .recover_interrupted_stream(
                    port.stream.clone(),
                    "crash".into(),
                    Arc::new(ElicitationOwnerLost),
                )
                .await
                .expect("recovery");
            let context = port
                .restore_context()
                .await
                .expect("restore")
                .expect("snapshot");
            assert_eq!(context.chat_projection().len(), 1);
            assert_eq!(context.chat_projection()[0].content, "Keep this request.");
        }
    }
}

#[tokio::test]
async fn crash_recovery_keeps_original_text_erasure_dependencies() {
    for backend in ["sqlite", "jsonl"] {
        for erase_before_recovery in [true, false] {
            let dir = tempfile::TempDir::new().expect("directory");
            let (port, authority) = session(dir.path(), backend);
            crash_with_text(&port, true, ProviderPurpose::Main, true).await;
            let history = port.owner.stream_history(&port.stream).expect("events");
            let mut target = None;
            for event in history {
                let EventData::ProviderResponseProgressObserved { observation_ref } =
                    event.body.data
                else {
                    continue;
                };
                let PayloadRead::Present(observation) = port
                    .owner
                    .store
                    .read_payload(&observation_ref.payload_id)
                    .expect("observation")
                else {
                    panic!("observation missing")
                };
                let observation: ProviderObservationPayload =
                    serde_json::from_slice(&observation.body).expect("decode");
                for operation in observation.operations {
                    if let ProviderObservation::ProtocolSegmentObserved {
                        component,
                        segment_ref,
                        ..
                    } = operation
                        && component == "text"
                    {
                        let PayloadRead::Present(block) = port
                            .owner
                            .store
                            .read_payload(&segment_ref.payload_id)
                            .expect("block")
                        else {
                            panic!("block missing")
                        };
                        if let ResponseBlock::Text { text_ref } =
                            serde_json::from_slice(&block.body).expect("block codec")
                        {
                            target = Some(text_ref);
                        }
                    }
                }
            }
            if !erase_before_recovery {
                port.owner
                    .recover_interrupted_stream(
                        port.stream.clone(),
                        "crash".into(),
                        Arc::new(ElicitationOwnerLost),
                    )
                    .await
                    .expect("recovery");
                assert_eq!(
                    port.restore_context()
                        .await
                        .expect("restore")
                        .expect("snapshot")
                        .chat_projection()
                        .len(),
                    2
                );
            }
            let intent = SessionEvent::typed(
                port.stream.clone(),
                EventData::PayloadErasureRequested {
                    targets: vec![target.expect("original text")],
                    reason_category: "operator_requested".into(),
                },
                port.actor.clone(),
            )
            .expect("intent");
            port.owner
                .admit_erasure(
                    PrincipalId::shared_operator(),
                    authority,
                    AppendRequest {
                        session_id: port.stream.clone(),
                        expected_head: port.owner.head(&port.stream).expect("head"),
                        idempotency_key: "erase-original".into(),
                        events: vec![intent],
                        payloads: vec![],
                        erasure_reference_commitment: None,
                    },
                )
                .await
                .expect("erase");
            if erase_before_recovery {
                port.owner
                    .recover_interrupted_stream(
                        port.stream.clone(),
                        "crash".into(),
                        Arc::new(ElicitationOwnerLost),
                    )
                    .await
                    .expect("settle without erased partial");
                assert_eq!(
                    port.restore_context()
                        .await
                        .expect("restore")
                        .expect("snapshot")
                        .chat_projection()
                        .len(),
                    1
                );
            } else {
                assert!(
                    port.restore_context().await.is_err(),
                    "recovered content cannot restore erased original text"
                );
            }
        }
    }
}

fn install_copy(target: &EngineSession, source: &EngineSession) {
    let input = source
        .owner
        .branch_context(&source.stream, &source.branch)
        .expect("source")
        .expect("context");
    let record = PayloadRecord::json(
        "session.context/v1",
        &serde_json::json!({
            "copy_source_session_id":source.stream, "copy_source_sequence":input.sequence,
            "copy_source_branch_id":source.branch,
        }),
    )
    .expect("copy");
    let mut projection = SessionEvent::typed(
        target.stream.clone(),
        EventData::ContextProjectionCreated {
            source_ranges: vec![],
            summary_ref: None,
            budget: 0,
            reason_key: "fixture_frozen_copy".into(),
            algorithm_version: "fixture/v1".into(),
            result_ref: StoredPayloadRef::from(&record),
            summary_source_refs: vec![],
            transform_started_event_id: None,
        },
        target.actor.clone(),
    )
    .expect("projection");
    projection.branch_id = Some(target.branch.clone());
    target
        .owner
        .append(AppendRequest {
            session_id: target.stream.clone(),
            expected_head: target.owner.head(&target.stream).expect("head"),
            idempotency_key: "copy".into(),
            events: vec![projection],
            payloads: vec![record],
            erasure_reference_commitment: None,
        })
        .expect("install copy");
}

#[tokio::test]
async fn crash_recovery_keeps_frozen_copy_prefix_and_branch_isolation() {
    for backend in ["sqlite", "jsonl"] {
        let dir = tempfile::TempDir::new().expect("directory");
        let (source, _) = session(dir.path(), backend);
        source
            .replace_context(
                ContextSnapshot {
                    messages: vec![ChatMessage::user("Keep this request.").into_conversation()],
                    history_has_trim_breadcrumb: None,
                },
                "source_history",
            )
            .await
            .expect("source context");
        let (port, _) = session(dir.path(), backend);
        let sibling = port
            .fork_branch(
                BranchId::new(),
                port.owner.head(&port.stream).expect("cursor"),
            )
            .await
            .expect("branch");
        crash_with_text(&port, true, ProviderPurpose::Main, true).await;
        install_copy(&port, &source);
        port.owner
            .recover_interrupted_stream(
                port.stream.clone(),
                "crash".into(),
                Arc::new(ElicitationOwnerLost),
            )
            .await
            .expect("recovery");
        let (copy, _) = session(dir.path(), backend);
        install_copy(&copy, &port);
        // A nested copy still resolves each inherited scope and its local
        // overlay. It does not flatten source authorization into the caller.
        for target in [&port, &copy] {
            let context = target
                .restore_context()
                .await
                .expect("copy restore")
                .expect("snapshot");
            assert_eq!(context.chat_projection().len(), 2);
            assert_eq!(context.chat_projection()[0].content, "Keep this request.");
            assert!(
                context.chat_projection()[1]
                    .content
                    .contains("Second unfinished\n\n[Interrupted:")
            );
        }
        assert!(
            sibling
                .restore_context()
                .await
                .expect("sibling")
                .expect("snapshot")
                .messages
                .is_empty(),
            "recovery must not leak into a sibling's frozen context"
        );
    }
}
