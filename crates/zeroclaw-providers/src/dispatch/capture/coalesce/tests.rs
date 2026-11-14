use super::*;
use zeroclaw_api::model_provider::{StreamChunk, StreamError};

#[tokio::test]
async fn compatible_attribution_emits_only_on_changes_without_dropping_text() {
    let mut body = String::new();
    for (model, text) in [
        (Some("first"), "a"),
        (Some("first"), "b"),
        (None, "c"),
        (Some("second"), "d"),
        (Some("first"), "e"),
    ] {
        let mut frame = serde_json::json!({"choices":[{"index":0,"delta":{"content":text}}]});
        if let Some(model) = model {
            frame["model"] = serde_json::json!(model);
        }
        body.push_str(&format!("data: {frame}\n\n"));
    }
    body.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
    let app = axum::Router::new().route(
        "/",
        axum::routing::get(move || async move { ([("content-type", "text/event-stream")], body) }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = zeroclaw_spawn::spawn!(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let response = reqwest::get(format!("http://{address}/")).await.unwrap();
    let events: Vec<_> = crate::compatible::sse_bytes_to_events(response, false)
        .collect()
        .await;
    server.abort();
    let mut models = Vec::new();
    let mut text = String::new();
    for event in events {
        match event.unwrap() {
            StreamEvent::ProtocolMetadata(metadata) if metadata["kind"] == "served_attribution" => {
                models.push(metadata["model"].as_str().unwrap().to_owned())
            }
            StreamEvent::TextDelta(chunk) => text.push_str(&chunk.delta),
            _ => {}
        }
    }
    assert_eq!(models, ["first", "second", "first"]);
    assert_eq!(text, "abcde");
}

fn text(text: &str) -> StreamEvent {
    StreamEvent::TextDelta(StreamChunk::delta(text).with_token_estimate())
}
fn position(output_index: u64) -> StreamEvent {
    StreamEvent::ContentPosition(ProviderContentPosition {
        output_index,
        content_index: None,
    })
}

#[tokio::test]
async fn batches_small_unicode_fragments_without_losing_token_counts() {
    let output: Vec<_> =
        stream(stream::iter([Ok(text("α")), Ok(text("β")), Ok(StreamEvent::Final)]).boxed())
            .collect()
            .await;
    assert_eq!(output.len(), 2);
    let StreamEvent::TextDelta(chunk) = output[0].as_ref().unwrap() else {
        panic!("text")
    };
    assert_eq!(chunk.delta, "αβ");
    assert_eq!(chunk.token_count, 2);
    assert!(matches!(output[1], Ok(StreamEvent::Final)));
}

#[tokio::test]
async fn batches_stay_bounded_and_flush_at_eof_and_before_error() {
    for error in [false, true] {
        let mut input: Vec<_> = (0..10_000)
            .map(|_| Ok(StreamEvent::ThinkingDelta("x".into())))
            .collect();
        if error {
            input.push(Err(StreamError::Http("fixture failure".into())));
        }
        let output: Vec<_> = stream(stream::iter(input).boxed()).collect().await;
        let mut bytes = 0;
        for item in &output {
            if let Ok(StreamEvent::ThinkingDelta(text)) = item {
                assert!(text.len() <= MAX_BYTES);
                bytes += text.len();
            }
        }
        assert_eq!(bytes, 10_000);
        assert!(output.len() <= 5);
        assert_eq!(output.last().unwrap().is_err(), error);
    }
}

#[tokio::test]
async fn protocol_positions_and_signed_reasoning_are_boundaries() {
    let input = vec![
        position(0),
        text("a"),
        position(0),
        text("b"),
        position(1),
        text("c"),
        StreamEvent::ReasoningFinalized("signed bytes".into()),
        StreamEvent::ThinkingDelta("thought".into()),
        text("answer"),
        StreamEvent::Final,
    ];
    let output: Vec<_> = stream(stream::iter(input.into_iter().map(Ok)).boxed())
        .collect()
        .await;
    assert_eq!(output.len(), 8);
    assert!(matches!(&output[0],Ok(StreamEvent::ContentPosition(p)) if p.output_index==0));
    assert!(matches!(&output[1],Ok(StreamEvent::TextDelta(c)) if c.delta=="ab"));
    assert!(matches!(&output[2],Ok(StreamEvent::ContentPosition(p)) if p.output_index==1));
    assert!(matches!(&output[3],Ok(StreamEvent::TextDelta(c)) if c.delta=="c"));
    assert!(matches!(&output[4],Ok(StreamEvent::ReasoningFinalized(s)) if s=="signed bytes"));
}

#[tokio::test]
async fn mixed_components_metadata_and_final_chunks_never_merge() {
    let mixed = StreamEvent::TextDelta(StreamChunk {
        delta: "answer".into(),
        reasoning: Some("thought".into()),
        is_final: false,
        token_count: 3,
    });
    let input = vec![
        text("a"),
        mixed,
        text("b"),
        StreamEvent::ProtocolMetadata(serde_json::json!({"kind":"finish"})),
        text("c"),
        StreamEvent::TextDelta(StreamChunk::final_chunk()),
    ];
    let output: Vec<_> = stream(stream::iter(input.into_iter().map(Ok)).boxed())
        .collect()
        .await;
    assert_eq!(output.len(), 6);
    assert!(
        matches!(&output[1], Ok(StreamEvent::TextDelta(c)) if c.reasoning.as_deref()==Some("thought") && c.delta=="answer")
    );
    assert!(matches!(&output[5], Ok(StreamEvent::TextDelta(c)) if c.is_final));
}

#[tokio::test(start_paused = true)]
async fn slow_stream_flushes_on_deadline_and_keeps_an_inflight_position() {
    let (sender, receiver) = tokio::sync::mpsc::channel(10);
    sender.send(Ok(position(0))).await.unwrap();
    sender.send(Ok(text("first"))).await.unwrap();
    sender.send(Ok(position(0))).await.unwrap();
    let mut output = stream(tokio_stream::wrappers::ReceiverStream::new(receiver).boxed());
    let started = tokio::time::Instant::now();
    assert!(matches!(
        output.next().await,
        Some(Ok(StreamEvent::ContentPosition(_)))
    ));
    assert_eq!(tokio::time::Instant::now() - started, MAX_DELAY);
    assert!(matches!(output.next().await,Some(Ok(StreamEvent::TextDelta(c))) if c.delta=="first"));
    sender.send(Ok(text("second"))).await.unwrap();
    sender.send(Ok(StreamEvent::Final)).await.unwrap();
    assert!(
        matches!(output.next().await,Some(Ok(StreamEvent::ContentPosition(p))) if p.output_index==0)
    );
    assert!(matches!(output.next().await,Some(Ok(StreamEvent::TextDelta(c))) if c.delta=="second"));
    assert!(matches!(output.next().await, Some(Ok(StreamEvent::Final))));
}

#[tokio::test(start_paused = true)]
async fn steady_fragments_do_not_extend_the_batch_deadline() {
    let input = stream::unfold(0, |n| async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        Some((Ok(text("x")), n + 1))
    })
    .boxed();
    let mut output = stream(input);
    let started = tokio::time::Instant::now();
    assert!(matches!(
        output.next().await,
        Some(Ok(StreamEvent::TextDelta(_)))
    ));
    assert_eq!(
        tokio::time::Instant::now() - started,
        Duration::from_millis(20) + MAX_DELAY
    );
}
