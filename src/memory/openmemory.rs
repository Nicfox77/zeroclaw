//! OpenMemory backend for ZeroClaw.
//!
//! Provides integration with OpenMemory cognitive memory engine via HTTP API.
//! Supports sectors (episodic, semantic, procedural, emotional, reflective),
//! salience scoring, decay, and waypoint graph.

use super::traits::{Memory, MemoryCategory, MemoryEntry};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use uuid::Uuid;

/// OpenMemory HTTP API backend.
///
/// Connects to an OpenMemory server (Docker or standalone) for cognitive
/// memory operations including sector classification, salience scoring,
/// decay management, and waypoint graph traversal.
pub struct OpenMemoryBackend {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    user_id: Option<String>,
    /// Tracks whether health check has been performed.
    initialized: OnceCell<()>,
}

/// Request body for /memory/add
#[derive(Serialize)]
struct AddMemoryRequest {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
}

/// Request body for /memory/query
#[derive(Serialize)]
struct QueryMemoryRequest {
    query: String,
    #[serde(default = "default_query_limit")]
    k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filters: Option<QueryFilters>,
}

fn default_query_limit() -> usize {
    8
}

#[derive(Serialize)]
struct QueryFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    sector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FlexibleTimestamp {
    String(String),
    I64(i64),
    U64(u64),
    F64(f64),
}

fn to_rfc3339_from_unix(value: i64) -> Option<String> {
    // Heuristic: 10+ digits are millis, lower magnitudes are seconds.
    let millis = if value.abs() >= 10_000_000_000 {
        value
    } else {
        value.saturating_mul(1000)
    };
    DateTime::<Utc>::from_timestamp_millis(millis).map(|dt| dt.to_rfc3339())
}

fn normalize_timestamp_text(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Utc::now().to_rfc3339();
    }

    if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
        return dt.with_timezone(&Utc).to_rfc3339();
    }

    if let Ok(value) = trimmed.parse::<i64>() {
        return to_rfc3339_from_unix(value).unwrap_or_else(|| Utc::now().to_rfc3339());
    }

    trimmed.to_string()
}

fn deserialize_optional_timestamp<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<FlexibleTimestamp>::deserialize(deserializer)?;

    let normalized = raw.map(|value| match value {
        FlexibleTimestamp::String(text) => normalize_timestamp_text(&text),
        FlexibleTimestamp::I64(value) => {
            to_rfc3339_from_unix(value).unwrap_or_else(|| Utc::now().to_rfc3339())
        }
        FlexibleTimestamp::U64(value) => i64::try_from(value)
            .ok()
            .and_then(to_rfc3339_from_unix)
            .unwrap_or_else(|| Utc::now().to_rfc3339()),
        FlexibleTimestamp::F64(value) => {
            if !value.is_finite() {
                Utc::now().to_rfc3339()
            } else {
                to_rfc3339_from_unix(value.round() as i64)
                    .unwrap_or_else(|| Utc::now().to_rfc3339())
            }
        }
    });

    Ok(normalized)
}


/// Response from /memory/add
#[derive(Deserialize)]
struct AddMemoryResponse {
    id: String,
    content: String,
    #[serde(default)]
    primary_sector: String,
    #[serde(default)]
    salience: f64,
}

/// Match item from /memory/query response
#[derive(Deserialize)]
struct MemoryMatch {
    id: String,
    content: String,
    score: f64,
    #[serde(default)]
    primary_sector: Option<String>,
    #[serde(default)]
    salience: Option<f64>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "deserialize_optional_timestamp")]
    created_at: Option<String>,
}

/// Response from /memory/query
#[derive(Deserialize)]
struct QueryMemoryResponse {
    #[serde(default)]
    matches: Vec<MemoryMatch>,
}

/// Response from /memory/all
#[derive(Deserialize)]
struct ListMemoryResponse {
    items: Vec<MemoryListItem>,
}

#[derive(Deserialize)]
struct MemoryListItem {
    id: String,
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    primary_sector: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_timestamp")]
    created_at: Option<String>,
    salience: Option<f64>,
}

/// Response from /memory/:id
#[derive(Deserialize)]
struct GetMemoryResponse {
    id: String,
    content: String,
    #[serde(default)]
    primary_sector: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "deserialize_optional_timestamp")]
    created_at: Option<String>,
    salience: Option<f64>,
}

impl OpenMemoryBackend {
    /// Create a new OpenMemory backend.
    ///
    /// # Arguments
    /// * `url` - OpenMemory server URL (e.g., "http://localhost:8080")
    /// * `api_key` - Optional API key for authentication
    /// * `user_id` - Optional user ID for multi-tenant isolation
    pub fn new(url: &str, api_key: Option<String>, user_id: Option<String>) -> Self {
        let base_url = url.trim_end_matches('/').to_string();
        let client = crate::config::build_runtime_proxy_client("memory.openmemory");

        Self {
            client,
            base_url,
            api_key,
            user_id,
            initialized: OnceCell::new(),
        }
    }

    /// Ensure the backend is healthy (called lazily on first operation).
    async fn ensure_initialized(&self) -> Result<()> {
        self.initialized
            .get_or_try_init(|| async {
                if !self.health_check().await {
                    anyhow::bail!("OpenMemory health check failed");
                }
                Ok::<(), anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);

        if let Some(ref key) = self.api_key {
            req = req.header("x-api-key", key);
        }

        req.header("Content-Type", "application/json")
    }

    async fn fetch_memory_detail(&self, id: &str) -> Option<GetMemoryResponse> {
        let mut request = self.request(reqwest::Method::GET, &format!("/memory/{}", id));
        if let Some(user_id) = self.user_id.as_deref() {
            request = request.query(&[("user_id", user_id)]);
        }

        let resp = request.send().await.ok()?;

        if resp.status() == StatusCode::NOT_FOUND {
            return None;
        }

        if !resp.status().is_success() {
            tracing::debug!(
                id = %id,
                status = %resp.status(),
                "OpenMemory detail fetch failed"
            );
            return None;
        }

        resp.json::<GetMemoryResponse>().await.ok()
    }

    fn extract_zeroclaw_key(metadata: Option<&serde_json::Value>) -> Option<String> {
        metadata
            .and_then(|m| m.get("zeroclaw_key"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn extract_zeroclaw_session_id(metadata: Option<&serde_json::Value>) -> Option<String> {
        metadata
            .and_then(|m| m.get("zeroclaw_session_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Map ZeroClaw MemoryCategory to OpenMemory sector.
    fn category_to_sector(category: &MemoryCategory) -> &'static str {
        match category {
            MemoryCategory::Core => "semantic",
            MemoryCategory::Daily => "episodic",
            MemoryCategory::Conversation => "episodic",
            MemoryCategory::Custom(name) => {
                // Map common custom categories to sectors
                match name.to_lowercase().as_str() {
                    "preference" | "preferences" => "semantic",
                    "event" | "events" => "episodic",
                    "skill" | "skills" | "procedure" => "procedural",
                    "emotion" | "emotional" | "feeling" => "emotional",
                    "reflection" | "insight" => "reflective",
                    _ => "semantic", // Default to semantic for unknown categories
                }
            }
        }
    }

    /// Map OpenMemory sector to ZeroClaw MemoryCategory.
    fn sector_to_category(sector: &str) -> MemoryCategory {
        match sector {
            "episodic" => MemoryCategory::Daily,
            "semantic" => MemoryCategory::Core,
            "procedural" => MemoryCategory::Custom("skill".to_string()),
            "emotional" => MemoryCategory::Custom("emotional".to_string()),
            "reflective" => MemoryCategory::Custom("reflection".to_string()),
            _ => MemoryCategory::Core,
        }
    }
}

#[async_trait]
impl Memory for OpenMemoryBackend {
    fn name(&self) -> &str {
        "openmemory"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> Result<()> {
        self.ensure_initialized().await?;

        let mapped_sector = Self::category_to_sector(&category);

        // Build metadata with ZeroClaw-specific fields.
        // IMPORTANT: do not set OpenMemory's reserved `metadata.sector` here,
        // otherwise it bypasses OpenMemory's own pattern-based classifier and
        // forces every memory to that sector.
        let metadata = serde_json::json!({
            "zeroclaw_key": key,
            "zeroclaw_turn_id": key,
            "zeroclaw_category": category.to_string(),
            "zeroclaw_session_id": session_id,
            "zeroclaw_context_id": session_id,
            "zeroclaw_mapped_sector": mapped_sector,
        });

        let category_label = category.to_string();
        let mut tags = vec![format!("zc:category:{}", category_label)];
        if matches!(&category, MemoryCategory::Conversation) {
            tags.push("zc:conversation".to_string());
        }

        let body = AddMemoryRequest {
            content: content.to_string(),
            tags: Some(tags),
            metadata: Some(metadata),
            user_id: self.user_id.clone(),
        };

        let resp = self
            .request(reqwest::Method::POST, "/memory/add")
            .json(&body)
            .send()
            .await
            .context("Failed to store memory in OpenMemory")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenMemory store failed ({}): {}", status, text);
        }

        tracing::debug!(
            key = %key,
            mapped_sector = %mapped_sector,
            "Stored memory in OpenMemory"
        );

        Ok(())
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        let filters = QueryFilters {
            sector: None,
            min_score: None,
            user_id: self.user_id.clone(),
        };

        let requested_limit = limit.max(1);
        // Keep OpenMemory as ranking authority. We only over-fetch when a strict
        // client-side session filter is requested.
        let fetch_limit = if session_id.is_some() {
            requested_limit.saturating_mul(4).clamp(requested_limit, 100)
        } else {
            requested_limit
        };

        let body = QueryMemoryRequest {
            query: query.to_string(),
            k: fetch_limit,
            user_id: self.user_id.clone(),
            filters: Some(filters),
        };

        let resp = self
            .request(reqwest::Method::POST, "/memory/query")
            .json(&body)
            .send()
            .await
            .context("Failed to query OpenMemory")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::warn!("OpenMemory query failed ({}): {}", status, text);
            return Ok(vec![]);
        }

        let result: QueryMemoryResponse = resp
            .json()
            .await
            .context("Failed to parse OpenMemory query response")?;

        let mut entries = Vec::with_capacity(requested_limit);
        for m in result.matches {
            let raw_score = if m.score.is_finite() {
                Some(m.score)
            } else {
                None
            };
            let mut key = Self::extract_zeroclaw_key(m.metadata.as_ref()).unwrap_or(m.id.clone());
            let mut recalled_session_id = Self::extract_zeroclaw_session_id(m.metadata.as_ref());
            let mut timestamp = m
                .created_at
                .clone()
                .unwrap_or_else(|| Utc::now().to_rfc3339());

            // Query responses can omit metadata depending on OpenMemory version.
            // Backfill missing fields from detail fetch only when needed.
            let needs_detail_fetch = key == m.id || recalled_session_id.is_none() || m.created_at.is_none();
            if needs_detail_fetch {
                if let Some(detail) = self.fetch_memory_detail(&m.id).await {
                    if key == m.id {
                        if let Some(stored_key) = Self::extract_zeroclaw_key(detail.metadata.as_ref()) {
                            key = stored_key;
                        }
                    }
                    if recalled_session_id.is_none() {
                        recalled_session_id = Self::extract_zeroclaw_session_id(detail.metadata.as_ref());
                    }
                    if m.created_at.is_none() {
                        if let Some(created_at) = detail.created_at {
                            timestamp = created_at;
                        }
                    }
                }
            }

            // Strict session isolation when a session_id is provided.
            if let Some(expected_session) = session_id {
                if recalled_session_id.as_deref() != Some(expected_session) {
                    continue;
                }
            }

            let category = m
                .primary_sector
                .as_deref()
                .map(|s| Self::sector_to_category(s))
                .unwrap_or(MemoryCategory::Core);

            entries.push(MemoryEntry {
                id: m.id,
                key,
                content: m.content,
                category,
                timestamp,
                session_id: recalled_session_id,
                score: raw_score,
            });

            if entries.len() >= requested_limit {
                break;
            }
        }

        Ok(entries)
    }

    async fn get(&self, key: &str) -> Result<Option<MemoryEntry>> {
        self.ensure_initialized().await?;

        // OpenMemory uses IDs, not keys. We need to search by metadata.
        // For now, we'll try to get by ID if key looks like a UUID.
        if let Ok(_) = Uuid::parse_str(key) {
            let mut request = self.request(reqwest::Method::GET, &format!("/memory/{}", key));
            if let Some(user_id) = self.user_id.as_deref() {
                request = request.query(&[("user_id", user_id)]);
            }

            let resp = request
                .send()
                .await
                .context("Failed to get memory from OpenMemory")?;

            if resp.status() == StatusCode::NOT_FOUND {
                return Ok(None);
            }

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("OpenMemory get failed ({}): {}", status, text);
            }

            let result: GetMemoryResponse = resp
                .json()
                .await
                .context("Failed to parse OpenMemory get response")?;

            // Extract ZeroClaw key from metadata if present
            let stored_key = result
                .metadata
                .as_ref()
                .and_then(|m| m.get("zeroclaw_key"))
                .and_then(|v| v.as_str())
                .unwrap_or(&result.id)
                .to_string();

            return Ok(Some(MemoryEntry {
                id: result.id,
                key: stored_key,
                content: result.content,
                category: result
                    .primary_sector
                    .as_deref()
                    .map(|s| Self::sector_to_category(s))
                    .unwrap_or(MemoryCategory::Core),
                timestamp: result.created_at.unwrap_or_else(|| Utc::now().to_rfc3339()),
                session_id: None,
                score: result.salience,
            }));
        }

        // Not a UUID, search by metadata key
        // OpenMemory doesn't have a direct key lookup, so we query
        Ok(None)
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        let mut url = format!("/memory/all?l=1000");

        if let Some(ref user_id) = self.user_id {
            url.push_str(&format!("&user_id={}", user_id));
        }

        // Filter by sector if category specified
        if let Some(cat) = category {
            let sector = Self::category_to_sector(cat);
            url.push_str(&format!("&sector={}", sector));
        }

        let resp = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .context("Failed to list memories from OpenMemory")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenMemory list failed ({}): {}", status, text);
        }

        let result: ListMemoryResponse = resp
            .json()
            .await
            .context("Failed to parse OpenMemory list response")?;

        let entries: Vec<MemoryEntry> = result
            .items
            .into_iter()
            .map(|m| {
                let stored_key = m
                    .tags
                    .iter()
                    .find(|t| t.starts_with("key:"))
                    .map(|t| t.strip_prefix("key:").unwrap_or(&m.id).to_string())
                    .unwrap_or_else(|| m.id.clone());

                MemoryEntry {
                    id: m.id.clone(),
                    key: stored_key,
                    content: m.content,
                    category: m
                        .primary_sector
                        .as_deref()
                        .map(|s| Self::sector_to_category(s))
                        .unwrap_or(MemoryCategory::Core),
                    timestamp: m.created_at.unwrap_or_else(|| Utc::now().to_rfc3339()),
                    session_id: None,
                    score: m.salience,
                }
            })
            .collect();

        Ok(entries)
    }

    async fn forget(&self, key: &str) -> Result<bool> {
        self.ensure_initialized().await?;

        // Try to delete by ID if key looks like a UUID
        if let Ok(_) = Uuid::parse_str(key) {
            let mut request = self.request(reqwest::Method::DELETE, &format!("/memory/{}", key));
            if let Some(user_id) = self.user_id.as_deref() {
                request = request.query(&[("user_id", user_id)]);
            }

            let resp = request
                .send()
                .await
                .context("Failed to delete memory from OpenMemory")?;

            if resp.status() == StatusCode::NOT_FOUND {
                return Ok(false);
            }

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("OpenMemory delete failed ({}): {}", status, text);
            }

            return Ok(true);
        }

        // Not a UUID, can't delete by key directly
        Ok(false)
    }

    async fn count(&self) -> Result<usize> {
        self.ensure_initialized().await?;

        let mut url = "/memory/all?l=1".to_string();
        if let Some(ref user_id) = self.user_id {
            url.push_str(&format!("&user_id={}", user_id));
        }

        let resp = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .context("Failed to count memories in OpenMemory")?;

        if !resp.status().is_success() {
            return Ok(0);
        }

        // OpenMemory doesn't return a count directly, so we'd need to count items
        // For now, return 0 and rely on health_check for connectivity
        Ok(0)
    }

    async fn health_check(&self) -> bool {
        let resp = self.request(reqwest::Method::GET, "/health").send().await;

        match resp {
            Ok(r) if r.status().is_success() => {
                tracing::info!("OpenMemory backend healthy at {}", self.base_url);
                true
            }
            Ok(r) => {
                tracing::warn!("OpenMemory health check failed: status {}", r.status());
                false
            }
            Err(e) => {
                tracing::warn!("OpenMemory health check failed: {}", e);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_to_sector_mapping() {
        assert_eq!(
            OpenMemoryBackend::category_to_sector(&MemoryCategory::Core),
            "semantic"
        );
        assert_eq!(
            OpenMemoryBackend::category_to_sector(&MemoryCategory::Daily),
            "episodic"
        );
        assert_eq!(
            OpenMemoryBackend::category_to_sector(&MemoryCategory::Conversation),
            "episodic"
        );
        assert_eq!(
            OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("skill".into())),
            "procedural"
        );
        assert_eq!(
            OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("emotional".into())),
            "emotional"
        );
        assert_eq!(
            OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("reflection".into())),
            "reflective"
        );
    }

    #[test]
    fn sector_to_category_mapping() {
        assert!(matches!(
            OpenMemoryBackend::sector_to_category("semantic"),
            MemoryCategory::Core
        ));
        assert!(matches!(
            OpenMemoryBackend::sector_to_category("episodic"),
            MemoryCategory::Daily
        ));
        assert!(matches!(
            OpenMemoryBackend::sector_to_category("procedural"),
            MemoryCategory::Custom(_)
        ));
    }

    #[test]
    fn backend_name() {
        let backend = OpenMemoryBackend::new("http://localhost:8080", None, None);
        assert_eq!(backend.name(), "openmemory");
    }
}
