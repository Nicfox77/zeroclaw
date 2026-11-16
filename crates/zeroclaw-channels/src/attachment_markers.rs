//! Pure display projection of explicit assistant attachment intent. Parsing
//! never opens a file or grants permission; callers admit effects only after
//! final assistant output and outbound policy have been accepted.
use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use std::ops::Range;

pub const MAX_FILES: usize = 16;
const PREFIX: &str = "<<file:";
const MAX_PATH_BYTES: usize = 4096;

pub const INSTRUCTIONS: &str = r#"## File Attachments

Attach an existing local file by wrapping its real path directly in <<file: and >> in the final answer. Prefer the exact absolute path returned by a file or image tool; copy it unchanged.
Workspace-relative example: <<file:images/portrait.png>> resolves images/portrait.png inside the current workspace root.
Absolute example: <<file:/workspace/images/portrait.png>> refers to that exact absolute path, subject to the configured file-read permissions.
These examples illustrate syntax; substitute the actual existing file path. The words workspace-relative are explanatory, never a directory prefix. Do not prepend workspace-relative/ or invent, add, remove, or rewrite path components. If the path is uncertain, verify it with file tools first.
Emit active attachment markers as plain text outside code blocks and quotes. Markers in code or quoted examples remain literal and do not attach files. Keep ordinary prose outside markers.
Use one marker per file, up to 16 files; multiple markers preserve their order. Images are detected from file contents; other files are sent as documents. Active markers are removed from visible text and speech.
No send_via call is needed for ordinary attachments; reserve that tool for routing, modality, or immediate sends. Do not repeat files already delivered by an immediate send. Use only existing local files, never URLs. Do not claim delivery of a missing or unsaved file.
"#;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Projection {
    pub text: String,
    pub paths: Vec<String>,
    pub invalid: bool,
}

fn literal_ranges(text: &str) -> Vec<Range<usize>> {
    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut stack = Vec::new();
    for (event, range) in Parser::new(text).into_offset_iter() {
        match event {
            Event::Start(
                tag @ (Tag::CodeBlock(_)
                | Tag::BlockQuote(_)
                | Tag::Link { .. }
                | Tag::Image { .. }),
            ) => {
                stack.push((tag.to_end(), range.start));
            }
            Event::End(
                end @ (TagEnd::CodeBlock | TagEnd::BlockQuote(_) | TagEnd::Link | TagEnd::Image),
            ) => {
                if let Some((expected, start)) = stack.pop()
                    && expected == end
                {
                    ranges.push(start..range.end);
                }
            }
            Event::Code(_) | Event::Html(_) | Event::InlineHtml(_) => ranges.push(range),
            _ => {}
        }
    }
    for (_, start) in stack {
        ranges.push(start..text.len());
    }
    ranges.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.start <= last.end
        {
            last.end = last.end.max(range.end);
            continue;
        }
        merged.push(range);
    }
    merged
}

/// Project a cumulative assistant frame. `finished=false` also withholds a
/// trailing partial marker prefix, so paths cannot flash into drafts or TTS
/// when delimiters arrive in different streamed fragments. Retain raw input as
/// the canonical accumulation; this result is a disposable view.
pub fn project(text: &str, finished: bool) -> Projection {
    let ranges = literal_ranges(text);
    let mut result = Projection::default();
    let mut i = 0;
    let mut literal = 0;
    while i < text.len() {
        while literal < ranges.len() && ranges[literal].end <= i {
            literal += 1;
        }
        if let Some(range) = ranges.get(literal).filter(|r| r.contains(&i)) {
            result.text.push_str(&text[i..range.end]);
            i = range.end;
            continue;
        }
        let rest = &text[i..];
        let escaped = text[..i].bytes().rev().take_while(|b| *b == b'\\').count() % 2 == 1;
        if !escaped
            && rest
                .get(..PREFIX.len())
                .is_some_and(|s| s.eq_ignore_ascii_case(PREFIX))
        {
            let Some(relative_end) = rest[PREFIX.len()..].find(">>") else {
                result.invalid = finished;
                break;
            };
            let end = i + PREFIX.len() + relative_end + 2;
            let quote = text[..i]
                .chars()
                .next_back()
                .filter(|c| matches!(c, '\'' | '"'));
            if quote.is_some_and(|q| text[end..].starts_with(q)) {
                result.text.push_str(&text[i..end]);
                i = end;
                continue;
            }
            let path = text[i + PREFIX.len()..end - 2].trim();
            if path.is_empty() || path.len() > MAX_PATH_BYTES || path.chars().any(char::is_control)
            {
                result.invalid = true;
            } else if !result.paths.iter().any(|p| p == path) {
                if result.paths.len() == MAX_FILES {
                    result.invalid = true;
                } else {
                    result.paths.push(path.to_owned());
                }
            }
            i = end;
            continue;
        }
        if !escaped && rest.len() < PREFIX.len() && PREFIX[..rest.len()].eq_ignore_ascii_case(rest)
        {
            if !finished {
                break;
            }
            if rest.len() > 2 {
                result.invalid = true;
                break;
            }
        }
        let ch = rest.chars().next().unwrap_or_default();
        result.text.push(ch);
        i += ch.len_utf8();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multiple_files_spaces_unicode_and_duplicates_preserve_order_and_prose() {
        let p = project(
            "Here are files.\n<<file:images/猫 photo.png>>\n<<FILE:report.pdf>>\n<<file:report.pdf>>\nDone.",
            true,
        );
        assert_eq!(p.paths, ["images/猫 photo.png", "report.pdf"]);
        assert_eq!(p.text, "Here are files.\n\n\n\nDone.");
        assert!(!p.invalid);
    }
    #[test]
    fn code_quotes_links_html_and_escapes_remain_literal() {
        for text in [
            "`<<file:a>>`",
            "``a ` <<file:a>>``",
            "```text\n<<file:a>>\n```",
            "~~~\n<<file:a>>\n~~~",
            "> <<file:a>>",
            "[<<file:a>>](https://example.com)",
            "<!-- <<file:a>> -->",
            "\\<<file:a>>",
            "\"<<file:a>>\"",
            "'<<file:a>>'",
        ] {
            let p = project(text, true);
            assert!(p.paths.is_empty(), "{text:?}: {p:?}");
            assert_eq!(p.text, text);
        }
    }
    #[test]
    fn every_stream_split_withholds_control_prefix_and_path_without_losing_following_text() {
        let text = "Hello. <<file:exports/猫 photo.png>> Finished.";
        for (i, _) in text.char_indices() {
            let p = project(&text[..i], false);
            assert!(!p.text.contains("<<"), "split {i}: {p:?}");
            assert!(!p.text.contains("exports/"), "split {i}: {p:?}");
        }
        assert_eq!(project(text, true).text, "Hello.  Finished.");
        assert_eq!(project("a <", true).text, "a <");
    }
    #[test]
    fn invalid_unclosed_and_over_limit_markers_never_expose_paths() {
        for text in [
            "Ready <<file:/private/path",
            "Ready <<fil",
            "Ready <<file:>>",
            "Ready <<file:a\nb>>",
        ] {
            let p = project(text, true);
            assert!(p.invalid, "{text:?}");
            assert!(!p.text.contains("<<"));
            assert!(!p.text.contains("/private/"));
        }
        let p = project(
            &format!("<<file:{}>>", "x".repeat(MAX_PATH_BYTES + 1)),
            true,
        );
        assert!(p.invalid);
        let p = project(
            &(0..=MAX_FILES)
                .map(|i| format!("<<file:{i}.txt>>"))
                .collect::<String>(),
            true,
        );
        assert!(p.invalid);
        assert_eq!(p.paths.len(), MAX_FILES);
    }
}
