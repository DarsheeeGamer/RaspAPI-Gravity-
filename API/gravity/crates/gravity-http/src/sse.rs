//! Line- and SSE-frame decoding over a byte stream.
//!
//! Providers consume streamed responses differently (Claude/ChatGPT use
//! `text/event-stream`; Gemini/Kiro use bespoke framing). This module offers the
//! shared lower layer — splitting a [`ByteStream`] into newline-delimited lines
//! without copying more than necessary — plus a small Server-Sent-Events parser
//! on top.

use crate::client::ByteStream;
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use gravity_core::Error;

/// Turn a byte stream into a stream of lines (newline stripped, no trailing
/// `\r`). The decoder buffers partial lines across chunks.
pub fn lines(mut stream: ByteStream) -> impl Stream<Item = Result<Bytes, Error>> {
    let mut buf = BytesMut::new();
    async_stream::try_stream! {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buf.extend_from_slice(&chunk);
            while let Some(pos) = memchr(b'\n', &buf) {
                let mut line = buf.split_to(pos + 1);
                // drop the '\n'
                line.truncate(line.len() - 1);
                // drop a trailing '\r'
                if line.last() == Some(&b'\r') {
                    line.truncate(line.len() - 1);
                }
                yield line.freeze();
            }
        }
        if !buf.is_empty() {
            yield buf.freeze();
        }
    }
}

/// A parsed Server-Sent Event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if present.
    pub event: Option<String>,
    /// The concatenated `data:` payload.
    pub data: String,
}

impl SseEvent {
    /// `true` when the data payload is the SSE terminator `[DONE]`.
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }
}

/// Parse an SSE byte stream into discrete [`SseEvent`]s.
///
/// Follows the SSE grammar: `data:`/`event:` fields accumulate until a blank
/// line dispatches the event. Comment lines (starting with `:`) are ignored.
pub fn events(stream: ByteStream) -> impl Stream<Item = Result<SseEvent, Error>> {
    let line_stream = lines(stream);
    async_stream::try_stream! {
        futures::pin_mut!(line_stream);
        let mut cur = SseEvent::default();
        let mut have_data = false;
        while let Some(line) = line_stream.next().await {
            let line = line?;
            if line.is_empty() {
                if have_data {
                    yield std::mem::take(&mut cur);
                    have_data = false;
                }
                continue;
            }
            if line.first() == Some(&b':') {
                continue; // comment / keep-alive
            }
            let (field, value) = split_field(&line);
            match field {
                b"data" => {
                    if have_data {
                        cur.data.push('\n');
                    }
                    cur.data.push_str(&String::from_utf8_lossy(value));
                    have_data = true;
                }
                b"event" => {
                    cur.event = Some(String::from_utf8_lossy(value).into_owned());
                }
                _ => {}
            }
        }
        if have_data {
            yield cur;
        }
    }
}

/// Split an SSE field line `field: value` into `(field, value)`, trimming a
/// single optional leading space from the value per the SSE spec.
fn split_field(line: &[u8]) -> (&[u8], &[u8]) {
    match memchr(b':', line) {
        Some(idx) => {
            let field = &line[..idx];
            let mut value = &line[idx + 1..];
            if value.first() == Some(&b' ') {
                value = &value[1..];
            }
            (field, value)
        }
        None => (line, &[]),
    }
}

/// Minimal `memchr` (avoids pulling the `memchr` crate for one call site).
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn byte_stream(chunks: Vec<&'static [u8]>) -> ByteStream {
        Box::pin(stream::iter(
            chunks.into_iter().map(|c| Ok(Bytes::from_static(c))),
        ))
    }

    #[tokio::test]
    async fn lines_split_across_chunks() {
        let s = byte_stream(vec![b"hel", b"lo\nwor", b"ld\n"]);
        let collected: Vec<_> = lines(s).collect().await;
        let got: Vec<String> = collected
            .into_iter()
            .map(|l| String::from_utf8(l.unwrap().to_vec()).unwrap())
            .collect();
        assert_eq!(got, vec!["hello", "world"]);
    }

    #[tokio::test]
    async fn sse_events_accumulate_and_dispatch() {
        let s = byte_stream(vec![
            b"event: message\ndata: {\"a\":1}\n\n",
            b": keepalive\ndata: [DONE]\n\n",
        ]);
        let collected: Vec<_> = events(s).collect().await;
        let evs: Vec<SseEvent> = collected.into_iter().map(Result::unwrap).collect();
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].event.as_deref(), Some("message"));
        assert_eq!(evs[0].data, r#"{"a":1}"#);
        assert!(evs[1].is_done());
    }
}
