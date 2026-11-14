//! Bound the durable block index by short semantic batches, not token frames.
//! Batches remain internal until surrounding capture commits them for delivery.
use super::*;
use std::{collections::VecDeque, time::Duration};
use zeroclaw_api::model_provider::ProviderContentPosition;

const MAX_BYTES: usize = 4096;
const MAX_DELAY: Duration = Duration::from_millis(100);

struct Segment {
    position: Option<ProviderContentPosition>,
    event: StreamResult<StreamEvent>,
}

struct State {
    inner: stream::BoxStream<'static, StreamResult<StreamEvent>>,
    // A position consumed while waiting on the next fragment must survive a
    // timer firing. These cursors own only not-yet-captured stream items.
    position: Option<ProviderContentPosition>,
    lookahead: Option<Segment>,
    ready: VecDeque<StreamResult<StreamEvent>>,
    ended: bool,
}

impl State {
    async fn next(&mut self) -> Option<Segment> {
        if let Some(segment) = self.lookahead.take() {
            return Some(segment);
        }
        if self.ended {
            return None;
        }
        loop {
            match self.inner.next().await {
                Some(Ok(StreamEvent::ContentPosition(position))) if self.position.is_none() => {
                    self.position = Some(position);
                }
                Some(event) => {
                    return Some(Segment {
                        position: self.position.take(),
                        event,
                    });
                }
                None => {
                    self.ended = true;
                    return self.position.take().map(|position| Segment {
                        position: None,
                        event: Ok(StreamEvent::ContentPosition(position)),
                    });
                }
            }
        }
    }
}

// Mixed text/reasoning chunks must stay separate: merging would reorder their
// alternating components. Finalized/signed blocks and all protocol metadata
// are barriers, even when their bytes happen to look like ordinary text.
fn fragment(event: &StreamEvent) -> Option<(u8, usize)> {
    match event {
        StreamEvent::ThinkingDelta(text) if !text.is_empty() => Some((0, text.len())),
        StreamEvent::TextDelta(chunk) if !chunk.is_final => {
            match (chunk.delta.is_empty(), chunk.reasoning.as_deref()) {
                (false, None | Some("")) => Some((1, chunk.delta.len())),
                (true, Some(reasoning)) if !reasoning.is_empty() => Some((2, reasoning.len())),
                _ => None,
            }
        }
        _ => None,
    }
}

fn append(current: &mut Segment, next: &Segment) -> bool {
    if current.position != next.position {
        return false;
    }
    let (Ok(current), Ok(next)) = (&mut current.event, &next.event) else {
        return false;
    };
    let (Some((kind, bytes)), Some((next_kind, next_bytes))) = (fragment(current), fragment(next))
    else {
        return false;
    };
    if kind != next_kind || bytes.saturating_add(next_bytes) > MAX_BYTES {
        return false;
    }
    match (current, next) {
        (StreamEvent::ThinkingDelta(current), StreamEvent::ThinkingDelta(next)) => {
            current.push_str(next);
        }
        (StreamEvent::TextDelta(current), StreamEvent::TextDelta(next)) => {
            let Some(tokens) = current.token_count.checked_add(next.token_count) else {
                return false;
            };
            current.delta.push_str(&next.delta);
            if let (Some(current), Some(next)) = (&mut current.reasoning, &next.reasoning) {
                current.push_str(next);
            }
            current.token_count = tokens;
        }
        _ => return false,
    }
    true
}

pub(super) fn stream(
    inner: stream::BoxStream<'static, StreamResult<StreamEvent>>,
) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
    let state = State {
        inner,
        position: None,
        lookahead: None,
        ready: VecDeque::new(),
        ended: false,
    };
    stream::unfold(state, |mut state| async move {
        if let Some(event) = state.ready.pop_front() {
            return Some((event, state));
        }
        let mut segment = state.next().await?;
        if segment.event.as_ref().ok().and_then(fragment).is_some() {
            // One deadline for the whole batch; a steady stream cannot keep
            // moving it forward and indefinitely delay durable/live progress.
            let deadline = tokio::time::Instant::now() + MAX_DELAY;
            loop {
                // A permanently ready upstream can beat timeout's timer poll.
                // Check the clock explicitly as well as bounding pending reads.
                if tokio::time::Instant::now() >= deadline {
                    break;
                }
                let Ok(Some(next)) = tokio::time::timeout_at(deadline, state.next()).await else {
                    break;
                };
                if !append(&mut segment, &next) {
                    state.lookahead = Some(next);
                    break;
                }
                if segment
                    .event
                    .as_ref()
                    .ok()
                    .and_then(fragment)
                    .is_some_and(|(_, n)| n >= MAX_BYTES)
                {
                    break;
                }
            }
        }
        if let Some(position) = segment.position {
            state
                .ready
                .push_back(Ok(StreamEvent::ContentPosition(position)));
        }
        state.ready.push_back(segment.event);
        state.ready.pop_front().map(|event| (event, state))
    })
    .boxed()
}

#[cfg(test)]
mod tests;
