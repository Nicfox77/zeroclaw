//! OpenMemory backend for ZeroClaw.
//!
//! Provides integration with OpenMemory cognitive memory engine via HTTP API.
//! Supports sectors (episodic, semantic, procedural, emotional, reflective),
//! salience scoring, decay, and waypoint graph.

use super::traits::{Memory, MemoryCategory, MemoryEntry};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Utc;
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

        let sector = Self::category_to_sector(&category);
        
        // Build metadata with ZeroClaw-specific fields
        let metadata = serde_json::json!({
            "zeroclaw_key": key,
            "zeroclaw_category": category.to_string(),
            "zeroclaw_session_id": session_id,
            "sector": sector,
        });

        let body = AddMemoryRequest {
            content: content.to_string(),
            tags: Some(vec![sector.to_string()]),
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
            sector = %sector,
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

        let body = QueryMemoryRequest {
            query: query.to_string(),
            k: limit,
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

        let entries: Vec<MemoryEntry> = result
            .matches
            .into_iter()
            .map(|m| MemoryEntry {
                id: m.id,
                key: String::new(), // OpenMemory doesn't use keys
                content: m.content,
                category: m
                    .primary_sector
                    .as_deref()
                    .map(|s| Self::sector_to_category(s))
                    .unwrap_or(MemoryCategory::Core),
                timestamp: Utc::now().to_rfc3339(),
                session_id: session_id.map(|s| s.to_string()),
                score: Some(m.score),
            })
            .collect();

        Ok(entries)
    }

    async fn get(&self, key: &str) -> Result<Option<MemoryEntry>> {
        self.ensure_initialized().await?;

        // OpenMemory uses IDs, not keys. We need to search by metadata.
        // For now, we'll try to get by ID if key looks like a UUID.
        if let Ok(_) = Uuid::parse_str(key) {
            let resp = self
                .request(reqwest::Method::GET, &format!("/memory/{}", key))
                .query(&[("user_id", self.user_id.as_deref())])
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
                let stored_key = m.tags
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
                    session_id: session_id.map(|s| s.to_string()),
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
            let resp = self
                .request(reqwest::Method::DELETE, &format!("/memory/{}", key))
                .query(&[("user_id", self.user_id.as_deref())])
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
        let resp = self
            .request(reqwest::Method::GET, "/health")
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                tracing::info!("OpenMemory backend healthy at {}", self.base_url);
                true
            }
            Ok(r) => {
                tracing::warn!(
                    "OpenMemory health check failed: status {}",
                    r.status()
                );
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
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Core), "semantic");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Daily), "episodic");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Conversation), "episodic");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("skill".into())), "procedural");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("emotional".into())), "emotional");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("reflection".into())), "reflective");
    }

    #[test]
    fn sector_to_category_mapping() {
        assert!(matches!(OpenMemoryBackend::sector_to_category("semantic"), MemoryCategory::Core));
        assert!(matches!(OpenMemoryBackend::sector_to_category("episodic"), MemoryCategory::Daily));
        assert!(matches!(OpenMemoryBackend::sector_to_category("procedural"), MemoryCategory::Custom(_)));
    }

    #[test]
    fn backend_name() {
        let backend = OpenMemoryBackend::new("http://localhost:8080", None, None);
        assert_eq!(backend.name(), "openmemory");
    }
}
