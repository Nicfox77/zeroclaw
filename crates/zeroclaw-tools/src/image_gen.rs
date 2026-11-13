use crate::helpers::{domain_guard, response_body};
use crate::i18n::{get_required_tool_string, get_required_tool_string_with_args};
use anyhow::Context;
use async_trait::async_trait;
use base64::Engine;
use reqwest::header::LOCATION;
use serde_json::json;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use zeroclaw_api::media::{MarkerKind, RenderedMarker};
use zeroclaw_api::tool::{Tool, ToolOutput, ToolResult, with_ephemeral_workspace_warning};
use zeroclaw_config::policy::SecurityPolicy;
use zeroclaw_config::policy::ToolOperation;
use zeroclaw_config::schema::OpenAIImageProviderConfig;

const FAL_RESPONSE_LIMIT_BYTES: usize = 1024 * 1024;
const FAL_ERROR_LIMIT_BYTES: usize = 16 * 1024;
const GENERATED_IMAGE_LIMIT_BYTES: usize = 20 * 1024 * 1024;
const MAX_IMAGE_REDIRECTS: usize = 10;
const OPENAI_RESPONSE_LIMIT_BYTES: usize = GENERATED_IMAGE_LIMIT_BYTES.div_ceil(3) * 4 + 64 * 1024;

struct OpenAIImageBackend {
    endpoint: reqwest::Url,
    extra_body: serde_json::Map<String, serde_json::Value>,
}

fn openai_image_size(size: &str) -> Option<String> {
    let size = match size {
        "square_hd" => "1024x1024",
        "landscape_4_3" => "1024x768",
        "portrait_4_3" => "768x1024",
        "landscape_16_9" => "1024x576",
        "portrait_16_9" => "576x1024",
        "auto" => return Some("auto".into()),
        value => value,
    };
    let (width, height) = size.split_once('x')?;
    if !width.bytes().all(|c| c.is_ascii_digit()) || !height.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (width, height) = (width.parse::<u32>().ok()?, height.parse::<u32>().ok()?);
    (width > 0 && height > 0).then(|| format!("{width}x{height}"))
}

async fn read_openai_image_body(response: reqwest::Response) -> anyhow::Result<Vec<u8>> {
    let payload = response_body::read_bounded(response, Some(OPENAI_RESPONSE_LIMIT_BYTES))
        .await
        .context(get_required_tool_string(
            "image-gen-error-endpoint-response",
        ))?;
    anyhow::ensure!(
        !payload.overflowed,
        get_required_tool_string("image-gen-error-json-limit")
    );
    let value: serde_json::Value = serde_json::from_slice(&payload.bytes).context(
        get_required_tool_string("image-gen-error-endpoint-response"),
    )?;
    let images = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .filter(|images| images.len() == 1)
        .ok_or_else(|| {
            anyhow::Error::msg(get_required_tool_string("image-gen-error-single-image"))
        })?;
    let encoded = images[0]
        .get("b64_json")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::Error::msg(get_required_tool_string("image-gen-error-base64")))?;
    anyhow::ensure!(
        encoded.len() <= GENERATED_IMAGE_LIMIT_BYTES.div_ceil(3) * 4,
        get_required_tool_string("image-gen-error-json-limit")
    );
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context(get_required_tool_string("image-gen-error-base64"))?;
    anyhow::ensure!(
        bytes.len() <= GENERATED_IMAGE_LIMIT_BYTES,
        get_required_tool_string("image-gen-error-json-limit")
    );
    anyhow::ensure!(
        bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        get_required_tool_string("image-gen-error-endpoint-png")
    );
    Ok(bytes)
}

struct ValidatedImageTarget {
    url: reqwest::Url,
    host: String,
    resolved_addrs: Vec<SocketAddr>,
}

fn parse_public_https_url(raw_url: &str) -> anyhow::Result<(reqwest::Url, String, u16)> {
    let raw_url = raw_url.trim();
    if raw_url.is_empty() || raw_url.chars().any(char::is_whitespace) {
        anyhow::bail!("Generated image URL must be a non-empty URL without whitespace");
    }

    let mut url = reqwest::Url::parse(raw_url).context("Invalid generated image URL")?;
    if url.scheme() != "https" {
        anyhow::bail!("Generated image URL must use HTTPS");
    }
    if !url.username().is_empty() || url.password().is_some() {
        anyhow::bail!("Generated image URL userinfo is not allowed");
    }

    let request_host = url
        .host_str()
        .ok_or_else(|| anyhow::Error::msg("Generated image URL must include a host"))?;
    if request_host.ends_with('.') {
        anyhow::bail!("Generated image URL host must not end with a dot");
    }
    let host = domain_guard::normalize_domain(request_host)
        .ok_or_else(|| anyhow::Error::msg("Generated image URL host is invalid"))?;
    let ip_literal = host.parse::<IpAddr>().ok();
    if domain_guard::is_private_or_local_host(&host) {
        anyhow::bail!("Generated image URL targets a local or non-global host");
    }
    if ip_literal.is_some_and(domain_guard::is_cloud_metadata_ip) {
        anyhow::bail!("Generated image URL targets a cloud metadata host");
    }

    // Request the host that was validated, not the raw spelling. Normalization
    // can change the host (a leading dot is stripped), and the DNS pin applied
    // by `generated_image_client_with_builder` is keyed on the normalized host.
    // Without this, `resolve_to_addrs` never matches the request host and
    // reqwest silently falls back to its own unvalidated lookup. IP-literal
    // hosts are left alone: they carry no DNS pin, and `set_host` rejects the
    // unbracketed IPv6 form that normalization produces.
    if ip_literal.is_none() {
        url.set_host(Some(&host))
            .map_err(|_| anyhow::Error::msg("Generated image URL host is invalid"))?;
    }

    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow::Error::msg("Generated image URL must include a valid port"))?;
    Ok((url, host, port))
}

async fn validate_image_target(
    raw_url: &str,
    nat64_prefixes: &[domain_guard::Nat64Prefix],
) -> anyhow::Result<ValidatedImageTarget> {
    let (url, host, port) = parse_public_https_url(raw_url)?;
    let resolved_addrs = if let Ok(ip) = host.parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        tokio::net::lookup_host((host.as_str(), port))
            .await
            .with_context(|| format!("Failed to resolve generated image host '{host}'"))?
            .collect::<Vec<_>>()
    };
    let ips = resolved_addrs
        .iter()
        .map(|addr| addr.ip())
        .collect::<Vec<_>>();
    domain_guard::validate_resolved_ips_are_public(&host, &ips, nat64_prefixes)?;

    Ok(ValidatedImageTarget {
        url,
        host,
        resolved_addrs,
    })
}

fn generated_image_client_with_builder(
    target: &ValidatedImageTarget,
    builder: reqwest::ClientBuilder,
) -> anyhow::Result<reqwest::Client> {
    let builder = builder
        .timeout(std::time::Duration::from_secs(120))
        .connect_timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy();
    let builder = if target.host.parse::<IpAddr>().is_ok() {
        builder
    } else {
        builder.resolve_to_addrs(&target.host, &target.resolved_addrs)
    };
    builder
        .build()
        .context("Failed to build generated image HTTP client")
}

fn generated_image_client(target: &ValidatedImageTarget) -> anyhow::Result<reqwest::Client> {
    generated_image_client_with_builder(target, reqwest::Client::builder())
}

async fn prepare_generated_image_target(
    raw_url: &str,
    nat64_prefixes: &[domain_guard::Nat64Prefix],
) -> anyhow::Result<(ValidatedImageTarget, reqwest::Client)> {
    let target = validate_image_target(raw_url, nat64_prefixes).await?;
    let client = generated_image_client(&target)?;
    Ok((target, client))
}

fn resolve_redirect_url(current: &reqwest::Url, location: &str) -> anyhow::Result<reqwest::Url> {
    current
        .join(location)
        .context("Invalid generated image redirect target")
}

fn resolve_image_filename(filename_arg: Option<&str>, nanos: u128) -> String {
    filename_arg
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            PathBuf::from(s).file_name().map_or_else(
                || "generated_image".to_string(),
                |n| n.to_string_lossy().to_string(),
            )
        })
        .unwrap_or_else(|| format!("generated_image_{nanos}"))
}

fn format_image_tool_output(
    path_display: &str,
    size_kb: usize,
    model: &str,
    prompt: &str,
) -> String {
    format!(
        "Image generated successfully.\n\
         File: {path_display}\n\
         Size: {size_kb} KB\n\
         Model: {model}\n\
         Prompt: {prompt}",
    )
}

async fn read_fal_success_body(response: reqwest::Response) -> anyhow::Result<Vec<u8>> {
    let body = response_body::read_bounded(response, Some(FAL_RESPONSE_LIMIT_BYTES))
        .await
        .context("Failed to read fal.ai response")?;
    if body.overflowed {
        anyhow::bail!("fal.ai response exceeds the 1 MiB size limit");
    }
    Ok(body.bytes)
}

async fn read_fal_error_text(response: reqwest::Response) -> anyhow::Result<String> {
    response_body::read_text(response, Some(FAL_ERROR_LIMIT_BYTES))
        .await
        .map(|(text, _)| text)
}

async fn read_generated_image_body(response: reqwest::Response) -> anyhow::Result<Vec<u8>> {
    let body = response_body::read_bounded(response, Some(GENERATED_IMAGE_LIMIT_BYTES))
        .await
        .context("Failed to read generated image bytes")?;
    if body.overflowed {
        anyhow::bail!(
            "Generated image exceeds the {} MiB size limit",
            GENERATED_IMAGE_LIMIT_BYTES / (1024 * 1024)
        );
    }
    Ok(body.bytes)
}

pub struct ImageGenTool {
    security: Arc<SecurityPolicy>,
    workspace_dir: PathBuf,
    default_model: String,
    api_key_env: String,
    openai: Option<OpenAIImageBackend>,
    api_key_file: Option<PathBuf>,
    persistent_writes: bool,
    nat64_prefixes: Vec<domain_guard::Nat64Prefix>,
}

impl ImageGenTool {
    pub fn new(
        security: Arc<SecurityPolicy>,
        workspace_dir: PathBuf,
        default_model: String,
        api_key_env: String,
        nat64_prefixes: Vec<String>,
    ) -> anyhow::Result<Self> {
        Self::new_with_persistence(
            security,
            workspace_dir,
            default_model,
            api_key_env,
            true,
            nat64_prefixes,
        )
    }

    /// Construct with an explicit persistence flag derived from the active
    /// runtime adapter's `has_filesystem_access()`. Mirrors
    /// [`super::file_write::FileWriteTool::new_with_persistence`].
    pub fn new_with_persistence(
        security: Arc<SecurityPolicy>,
        workspace_dir: PathBuf,
        default_model: String,
        api_key_env: String,
        persistent_writes: bool,
        nat64_prefixes: Vec<String>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            security,
            workspace_dir,
            default_model,
            api_key_env,
            openai: None,
            api_key_file: None,
            persistent_writes,
            nat64_prefixes: domain_guard::parse_nat64_prefixes(
                &nat64_prefixes,
                "security.nat64_prefixes",
            )?,
        })
    }

    /// Materialize the selected operator-owned provider profile. Neither the URI
    /// nor the credential source is accepted from model/tool arguments.
    pub fn with_openai_profile(
        mut self,
        profile: &OpenAIImageProviderConfig,
    ) -> anyhow::Result<Self> {
        profile.validate()?;
        let extra_body = profile
            .extra_body
            .as_ref()
            .map(OpenAIImageProviderConfig::validate_extra_body)
            .transpose()?
            .cloned()
            .unwrap_or_default();
        self.openai = Some(OpenAIImageBackend {
            endpoint: profile.generation_url()?,
            extra_body,
        });
        self.default_model = profile.model.clone();
        self.api_key_env = profile.api_key_env.clone();
        self.api_key_file = profile.api_key_file.clone();
        Ok(self)
    }

    async fn configured_api_key(&self) -> Result<String, String> {
        let Some(path) = &self.api_key_file else {
            return Self::read_api_key(&self.api_key_env);
        };
        let key = tokio::fs::read_to_string(path)
            .await
            .map_err(|_| get_required_tool_string("image-gen-error-key-read"))?;
        let key = key.trim();
        if key.is_empty() || key.len() > 16384 || key.chars().any(char::is_control) {
            return Err(get_required_tool_string("image-gen-error-key-invalid"));
        }
        Ok(key.to_owned())
    }

    async fn generate_openai(
        &self,
        backend: &OpenAIImageBackend,
        api_key: &str,
        model: &str,
        prompt: &str,
        size: &str,
        extra_body: Option<&serde_json::Value>,
    ) -> anyhow::Result<Vec<u8>> {
        let size = openai_image_size(size)
            .ok_or_else(|| anyhow::Error::msg(get_required_tool_string("image-gen-error-size")))?;
        let mut body = backend.extra_body.clone();
        if let Some(extra) = extra_body {
            body.extend(
                OpenAIImageProviderConfig::validate_extra_body(extra)
                    .context(get_required_tool_string("image-gen-error-extras"))?
                    .clone(),
            );
        }
        body.extend(serde_json::from_value::<
            serde_json::Map<String, serde_json::Value>,
        >(
            json!({"model":model,"prompt":prompt,"size":size,"n":1})
        )?);
        let response = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()?
            .post(backend.endpoint.clone())
            .bearer_auth(api_key)
            .header("Accept", "application/json")
            .json(&body)
            .send()
            .await
            .context(get_required_tool_string("image-gen-error-endpoint-request"))?;
        anyhow::ensure!(
            response.status().is_success(),
            get_required_tool_string_with_args(
                "image-gen-error-endpoint-status",
                &[("status", &response.status().to_string())]
            )
        );
        read_openai_image_body(response).await
    }

    /// Build a reusable HTTP client with reasonable timeouts.
    fn http_client() -> anyhow::Result<reqwest::Client> {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("Failed to build fal.ai HTTP client")
    }

    async fn download_generated_image(
        image_url: &str,
        nat64_prefixes: &[domain_guard::Nat64Prefix],
    ) -> anyhow::Result<Vec<u8>> {
        let mut current_url =
            reqwest::Url::parse(image_url).context("Invalid generated image URL")?;

        for redirect_count in 0..=MAX_IMAGE_REDIRECTS {
            let (target, client) =
                prepare_generated_image_target(current_url.as_str(), nat64_prefixes).await?;
            let response = client
                .get(target.url.clone())
                .send()
                .await
                .context("Failed to download generated image")?;

            if response.status().is_redirection() {
                if redirect_count == MAX_IMAGE_REDIRECTS {
                    anyhow::bail!("Too many generated image redirects (max {MAX_IMAGE_REDIRECTS})");
                }
                let location = response
                    .headers()
                    .get(LOCATION)
                    .ok_or_else(|| anyhow::Error::msg("Generated image redirect omitted Location"))?
                    .to_str()
                    .context("Generated image redirect Location is not valid text")?;
                current_url = resolve_redirect_url(&target.url, location)?;
                continue;
            }

            if !response.status().is_success() {
                anyhow::bail!(
                    "Generated image download failed with HTTP {}",
                    response.status()
                );
            }

            return read_generated_image_body(response).await;
        }

        anyhow::bail!("Generated image redirect limit exceeded")
    }

    /// Read an API key from the environment.
    fn read_api_key(env_var: &str) -> Result<String, String> {
        std::env::var(env_var)
            .map(|v| v.trim().to_string())
            .ok()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("Missing API key: set the {env_var} environment variable"))
    }

    /// Core generation logic: call fal.ai, download image, save to disk.
    async fn generate(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // ── Parse parameters ───────────────────────────────────────
        let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => p.trim().to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some("Missing required parameter: 'prompt'".into()),
                });
            }
        };

        // Sanitize filename — strip path components to prevent traversal.
        // When the caller doesn't provide one, generate a unique default so
        // successive calls without an explicit name never clobber each other.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let safe_name =
            resolve_image_filename(args.get("filename").and_then(|v| v.as_str()), nanos);

        let size = args
            .get("size")
            .and_then(|v| v.as_str())
            .unwrap_or("square_hd");

        // Validate size enum.
        const VALID_SIZES: &[&str] = &[
            "square_hd",
            "landscape_4_3",
            "portrait_4_3",
            "landscape_16_9",
            "portrait_16_9",
        ];
        if self.openai.is_none() && !VALID_SIZES.contains(&size) {
            return Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(format!(
                    "Invalid size '{size}'. Valid values: {}",
                    VALID_SIZES.join(", ")
                )),
            });
        }

        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(&self.default_model);

        // Validate model identifier: must look like a fal.ai model path
        // (e.g. "fal-ai/flux/schnell"). Reject values with "..", query
        // strings, or fragments that could redirect the HTTP request.
        if self.openai.is_none()
            && (model.contains("..")
                || model.contains('?')
                || model.contains('#')
                || model.contains('\\')
                || model.starts_with('/'))
        {
            return Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(format!(
                    "Invalid model identifier '{model}'. \
                     Must be a fal.ai model path (e.g. 'fal-ai/flux/schnell')."
                )),
            });
        }

        // ── Read API key ───────────────────────────────────────────
        let api_key = match self.configured_api_key().await {
            Ok(k) => k,
            Err(msg) => {
                return Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some(msg),
                });
            }
        };

        let bytes = if let Some(backend) = &self.openai {
            self.generate_openai(
                backend,
                &api_key,
                model,
                &prompt,
                size,
                args.get("extra_body"),
            )
            .await?
        } else {
            // ── Call fal.ai ────────────────────────────────────────────
            let client = Self::http_client()?;
            let url = format!("https://fal.run/{model}");

            let body = json!({
                "prompt": prompt,
                "image_size": size,
                "num_images": 1
            });

            let resp = client
                .post(&url)
                .header("Authorization", format!("Key {api_key}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .context("fal.ai request failed")?;

            let status = resp.status();
            if !status.is_success() {
                let body_text = read_fal_error_text(resp).await.unwrap_or_default();
                return Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some(format!("fal.ai API error ({status}): {body_text}")),
                });
            }

            let fal_body = read_fal_success_body(resp).await?;
            let resp_json: serde_json::Value = serde_json::from_slice(&fal_body)
                .context("Failed to parse fal.ai response as JSON")?;

            let image_url = resp_json
                .pointer("/images/0/url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ::zeroclaw_log::record!(
                        ERROR,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure),
                        "image_gen: fal.ai response missing image URL"
                    );
                    anyhow::Error::msg("No image URL in fal.ai response")
                })?;

            // ── Download image ─────────────────────────────────────────
            match Self::download_generated_image(image_url, &self.nat64_prefixes).await {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(ToolResult {
                        success: false,
                        output: ToolOutput::default(),
                        error: Some(error.to_string()),
                    });
                }
            }
        };

        // ── Save to disk ───────────────────────────────────────────
        let images_dir = self.workspace_dir.join("images");
        tokio::fs::create_dir_all(&images_dir)
            .await
            .context("Failed to create images directory")?;

        let output_path = images_dir.join(format!("{safe_name}.png"));
        tokio::fs::write(&output_path, &bytes)
            .await
            .context("Failed to write image file")?;

        let size_kb = bytes.len() / 1024;

        let path_display = output_path.display().to_string();
        let output = format_image_tool_output(&path_display, size_kb, model, &prompt);

        // The generated image is declared as an attachment; the body keeps
        // the durable File: line instead of marker syntax (which is text
        // under the attachment-identity contract and would never promote).
        Ok(ToolResult {
            success: true,
            output: output.into(),
            error: None,
        }
        .with_attachment(RenderedMarker {
            target: path_display,
            kind: MarkerKind::Image,
        }))
    }
}

#[async_trait]
impl Tool for ImageGenTool {
    fn name(&self) -> &str {
        "image_gen"
    }

    fn description(&self) -> &str {
        "Generate an image from a text prompt using the configured image model. \
         Saves the result to the workspace images directory and returns the file path."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let mut schema = json!({
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the image to generate."
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename without extension (default: 'generated_image'). Saved as PNG in workspace/images/."
                },
                "size": {
                    "type": "string",
                    "enum": ["square_hd", "landscape_4_3", "portrait_4_3", "landscape_16_9", "portrait_16_9"],
                    "description": "Image aspect ratio / size preset (default: 'square_hd')."
                },
                "model": {
                    "type": "string",
                    "description": get_required_tool_string("image-gen-model-description")
                }
            }
        });
        if self.openai.is_some() {
            schema["properties"]["size"] = json!({"type":"string", "description":get_required_tool_string("image-gen-size-description")});
            schema["properties"]["extra_body"] = json!({"type":"object", "description":get_required_tool_string("image-gen-extras-description")});
        }
        schema
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // Security: image generation is a side-effecting action (HTTP + file write).
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "image_gen")
        {
            return Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(error),
            });
        }

        let result = self.generate(args).await?;
        Ok(warn_if_ephemeral(self.persistent_writes, result))
    }
}

/// A generated image saved to an ephemeral workspace never reaches the host
/// and is lost at session end; warn loudly on success. The warning rewrites
/// the text only: the declared image attachment must survive it.
fn warn_if_ephemeral(persistent_writes: bool, mut result: ToolResult) -> ToolResult {
    if !persistent_writes && result.success {
        result.output.map_text(with_ephemeral_workspace_warning);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use zeroclaw_config::autonomy::AutonomyLevel;
    use zeroclaw_config::policy::SecurityPolicy;

    fn openai_profile(uri: &str, key_file: Option<PathBuf>) -> OpenAIImageProviderConfig {
        OpenAIImageProviderConfig {
            uri: uri.into(),
            model: "example/image-model".into(),
            api_key_file: key_file,
            ..Default::default()
        }
    }
    fn png_fixture() -> Vec<u8> {
        base64::engine::general_purpose::STANDARD.decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0qkAAAAASUVORK5CYII=").unwrap()
    }
    #[test]
    fn openai_image_provider_rejects_unsafe_transport_and_embedded_credentials() {
        for uri in [
            "http://example.invalid/v1",
            "ftp://127.0.0.1/v1",
            "https://user:password@example.invalid/v1",
            "https://example.invalid/v1?key=value",
            "https://example.invalid/v1#fragment",
            " http://127.0.0.1/v1",
        ] {
            assert!(
                test_tool()
                    .with_openai_profile(&openai_profile(uri, None))
                    .is_err(),
                "{uri}"
            );
        }
        for uri in [
            "http://127.0.0.1:1234/v1",
            "http://[::1]:1234/v1",
            "https://example.invalid/prefix/v1/",
        ] {
            let tool = test_tool()
                .with_openai_profile(&openai_profile(uri, None))
                .unwrap();
            assert!(
                tool.openai
                    .as_ref()
                    .unwrap()
                    .endpoint
                    .path()
                    .ends_with("/v1/images/generations")
            );
        }
        assert!(
            test_tool()
                .with_openai_profile(&openai_profile(
                    "http://127.0.0.1/v1",
                    Some("relative.key".into())
                ))
                .is_err()
        );
    }
    #[tokio::test]
    async fn openai_image_provider_generates_attachment_rereads_key_and_merges_extras() {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{body_json, header, method, path},
        };
        let server = MockServer::start().await;
        let workspace = tempfile::tempdir().unwrap();
        let key_file = workspace.path().join("api.key");
        let mut profile = openai_profile(&format!("{}/v1", server.uri()), Some(key_file.clone()));
        profile.extra_body = Some(json!({"seed":7,"steps":4}));
        let mut tool = test_tool();
        tool.workspace_dir = workspace.path().into();
        let tool = tool.with_openai_profile(&profile).unwrap();
        let bytes = png_fixture();
        for (index, key) in ["fixture-key-one", "fixture-key-two"].iter().enumerate() {
            tokio::fs::write(&key_file, format!("{key}\n"))
                .await
                .unwrap();
            Mock::given(method("POST")).and(path("/v1/images/generations"))
                .and(header("authorization", format!("Bearer {key}")))
                .and(body_json(json!({"model":"example/image-model", "prompt":"A blue circle", "n":1,
                    "size":"768x1024", "seed":index, "steps":4, "negative_prompt":"text"})))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":[{"b64_json":base64::engine::general_purpose::STANDARD.encode(&bytes)}]})))
                .expect(1).mount(&server).await;
            let result = tool.execute(json!({"prompt":"A blue circle", "size":"768x1024", "filename":format!("fixture-{index}"),
                "extra_body":{"seed":index,"negative_prompt":"text"}})).await.unwrap();
            assert!(result.success);
            assert_eq!(result.output.attachments().len(), 1);
            let file = workspace.path().join(format!("images/fixture-{index}.png"));
            assert_eq!(tokio::fs::read(&file).await.unwrap(), bytes);
            assert_eq!(
                result.output.attachments()[0].target,
                file.display().to_string()
            );
        }
    }
    #[tokio::test]
    async fn openai_image_provider_rejects_redirects_error_bodies_and_malformed_results() {
        use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
        let workspace = tempfile::tempdir().unwrap();
        let key = workspace.path().join("api.key");
        tokio::fs::write(&key, "fixture-private-key").await.unwrap();
        let valid =
            json!({"b64_json":base64::engine::general_purpose::STANDARD.encode(png_fixture())});
        for (status, body) in [(302, "redirect".into()), (401, "fixture-private-key".into()),
            (200, "not JSON".into()), (200, json!({"data":[]}).to_string()),
            (200, json!({"data":[{"url":"http://127.0.0.1:1/must-not-fetch"}]}).to_string()),
            (200, json!({"data":[{"b64_json":"not-base64!"}]}).to_string()),
            (200, json!({"data":[valid.clone(),valid]}).to_string()),
            (200, json!({"data":[{"b64_json":base64::engine::general_purpose::STANDARD.encode("not PNG")}]}).to_string())] {
            let server = MockServer::start().await;
            Mock::given(method("POST")).respond_with(ResponseTemplate::new(status)
                .insert_header("Location", "http://127.0.0.1:1/must-not-follow").set_body_string(body))
                .expect(1).mount(&server).await;
            let mut tool = test_tool(); tool.workspace_dir = workspace.path().into();
            let tool = tool.with_openai_profile(&openai_profile(&server.uri(), Some(key.clone()))).unwrap();
            let error = tool.execute(json!({"prompt":"A blue circle"})).await.unwrap_err();
            assert!(!error.to_string().contains("fixture-private-key"));
            assert!(!workspace.path().join("images").exists());
        }
    }
    #[tokio::test]
    async fn openai_image_provider_rejects_reserved_extras_before_request() {
        use wiremock::MockServer;
        let workspace = tempfile::tempdir().unwrap();
        let key = workspace.path().join("api.key");
        tokio::fs::write(&key, "fixture-key").await.unwrap();
        let server = MockServer::start().await;
        let profile = openai_profile(&server.uri(), Some(key));
        let tool = test_tool().with_openai_profile(&profile).unwrap();
        for extra in [
            json!({"model":"other"}),
            json!({"n":2}),
            json!({"stream":true}),
            json!({"size":"512x512"}),
            json!({"api_key":"other"}),
            json!({"response_format":"url"}),
            json!({"output_format":"jpeg"}),
            json!("invalid"),
        ] {
            assert!(
                tool.execute(json!({"prompt":"A circle","extra_body":extra}))
                    .await
                    .is_err()
            );
            let mut invalid = profile.clone();
            invalid.extra_body = Some(extra);
            assert!(test_tool().with_openai_profile(&invalid).is_err());
        }
        assert!(server.received_requests().await.unwrap().is_empty());
    }
    #[tokio::test]
    async fn openai_image_key_file_failure_does_not_fall_back_to_environment() {
        let workspace = tempfile::tempdir().unwrap();
        let key = workspace.path().join("missing.key");
        let tool = test_tool()
            .with_openai_profile(&openai_profile("http://127.0.0.1:1/v1", Some(key.clone())))
            .unwrap();
        assert!(
            tool.configured_api_key()
                .await
                .unwrap_err()
                .contains("read")
        );
        tokio::fs::write(&key, "\n").await.unwrap();
        assert!(
            tool.configured_api_key()
                .await
                .unwrap_err()
                .contains("empty")
        );
    }
    #[test]
    fn openai_image_sizes_preserve_rectangles_and_validate_dimensions() {
        assert_eq!(openai_image_size("512x512").as_deref(), Some("512x512"));
        assert_eq!(openai_image_size("1280x720").as_deref(), Some("1280x720"));
        assert_eq!(
            openai_image_size("landscape_16_9").as_deref(),
            Some("1024x576")
        );
        assert_eq!(openai_image_size("auto").as_deref(), Some("auto"));
        for invalid in [
            "0x512",
            "512x0",
            "-1x512",
            "+1x512",
            "512X512",
            "512x512x2",
            "invalid",
            "4294967296x1",
        ] {
            assert!(openai_image_size(invalid).is_none());
        }
    }
    #[tokio::test]
    async fn openai_image_response_is_bounded_without_waiting_for_eof() {
        let (response, release, server) =
            held_open_chunked_response(OPENAI_RESPONSE_LIMIT_BYTES + 1).await;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            read_openai_image_body(response),
        )
        .await;
        finish_held_open_response(release, server).await;
        assert!(
            result
                .expect("reader waited for EOF")
                .unwrap_err()
                .to_string()
                .contains("limit")
        );
    }

    async fn held_open_chunked_response(
        byte_count: usize,
    ) -> (
        reqwest::Response,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();

        let server = zeroclaw_spawn::spawn!(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            let chunk = vec![b'x'; 64 * 1024];
            let mut remaining = byte_count;
            while remaining > 0 {
                let chunk_len = remaining.min(chunk.len());
                stream
                    .write_all(format!("{chunk_len:x}\r\n").as_bytes())
                    .await
                    .unwrap();
                stream.write_all(&chunk[..chunk_len]).await.unwrap();
                stream.write_all(b"\r\n").await.unwrap();
                remaining -= chunk_len;
            }
            let _ = release_rx.await;
            let _ = stream.write_all(b"0\r\n\r\n").await;
        });

        (
            reqwest::get(format!("http://{addr}")).await.unwrap(),
            release_tx,
            server,
        )
    }

    async fn finish_held_open_response(
        release: tokio::sync::oneshot::Sender<()>,
        server: tokio::task::JoinHandle<()>,
    ) {
        let _ = release.send(());
        server.await.unwrap();
    }

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        })
    }

    fn test_tool() -> ImageGenTool {
        ImageGenTool::new(
            test_security(),
            std::env::temp_dir(),
            "fal-ai/flux/schnell".into(),
            "FAL_API_KEY".into(),
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn generated_image_target_rejects_private_and_userinfo_urls() {
        for url in [
            "https://127.0.0.1/image.png",
            "https://169.254.169.254/latest/meta-data",
            "https://user@example.com/image.png",
        ] {
            assert!(parse_public_https_url(url).is_err(), "accepted {url}");
        }
    }

    #[tokio::test]
    async fn generated_image_redirect_to_private_target_is_rejected() {
        let current = reqwest::Url::parse("https://cdn.example.com/image.png").unwrap();
        let redirect = resolve_redirect_url(&current, "https://127.0.0.1/private.png").unwrap();

        assert!(
            prepare_generated_image_target(redirect.as_str(), &[])
                .await
                .is_err()
        );
    }

    #[test]
    fn generated_image_target_requires_https() {
        assert!(parse_public_https_url("http://example.com/image.png").is_err());
        assert!(parse_public_https_url("https://example.com/image.png").is_ok());
    }

    #[test]
    fn generated_image_target_rejects_trailing_dot_host() {
        assert!(parse_public_https_url("https://example.com./image.png").is_err());
    }

    /// The returned URL must carry the host that was actually validated, not
    /// the raw spelling. `normalize_domain` strips a leading dot, so
    /// `https://.example.com/` validates and pins `example.com` while the raw
    /// URL still asks for `.example.com`. The pin is keyed on the normalized
    /// host, so a mismatch means `resolve_to_addrs` never matches and reqwest
    /// performs its own unvalidated lookup for the request host.
    #[test]
    fn generated_image_target_url_host_matches_the_validated_host() {
        let (url, host, _port) = parse_public_https_url("https://.example.com/img.png")
            .expect("leading-dot host normalizes to a public host");

        assert_eq!(host, "example.com", "host must normalize");
        assert_eq!(
            url.host_str(),
            Some(host.as_str()),
            "returned URL must request the validated host, otherwise the DNS \
             pin keyed on '{host}' never matches and reqwest re-resolves \
             '{:?}' unvalidated",
            url.host_str()
        );
    }

    #[tokio::test]
    async fn generated_image_target_accepts_public_ipv6_literal_without_dns() {
        let (target, _client) =
            prepare_generated_image_target("https://[2606:4700:4700::1111]/image.png", &[])
                .await
                .unwrap();
        let expected_ip = "2606:4700:4700::1111".parse::<IpAddr>().unwrap();

        assert_eq!(target.host, expected_ip.to_string());
        assert_eq!(
            target.resolved_addrs,
            vec![SocketAddr::new(expected_ip, 443)]
        );
    }

    #[tokio::test]
    async fn generated_image_client_clears_proxy_configuration() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let direct_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let direct_addr = direct_listener.local_addr().unwrap();
        let proxy_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();

        let direct_task = zeroclaw_spawn::spawn!(async move {
            let (mut stream, _) = direct_listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\ndirect",
                )
                .await
                .unwrap();
        });
        let proxy_task = zeroclaw_spawn::spawn!(async move {
            let (mut stream, _) = proxy_listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nproxy",
                )
                .await
                .unwrap();
        });

        let target = ValidatedImageTarget {
            url: reqwest::Url::parse(&format!("http://image.test:{}/", direct_addr.port()))
                .unwrap(),
            host: "image.test".to_string(),
            resolved_addrs: vec![direct_addr],
        };
        let proxy = reqwest::Proxy::http(format!("http://{proxy_addr}")).unwrap();
        let proxied_response = reqwest::Client::builder()
            .proxy(proxy.clone())
            .resolve_to_addrs(&target.host, &target.resolved_addrs)
            .build()
            .unwrap()
            .get(target.url.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(proxied_response.text().await.unwrap(), "proxy");
        proxy_task.await.unwrap();

        let direct_response =
            generated_image_client_with_builder(&target, reqwest::Client::builder().proxy(proxy))
                .unwrap()
                .get(target.url.clone())
                .send()
                .await
                .unwrap();

        assert_eq!(direct_response.text().await.unwrap(), "direct");
        direct_task.await.unwrap();
    }

    #[tokio::test]
    async fn generated_image_body_rejects_oversized_chunked_response() {
        let (response, release, server) =
            held_open_chunked_response(GENERATED_IMAGE_LIMIT_BYTES + 1).await;

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            read_generated_image_body(response),
        )
        .await;
        finish_held_open_response(release, server).await;
        let error = result
            .expect("generated image reader waited for EOF after crossing its limit")
            .unwrap_err();

        assert!(error.to_string().contains("20 MiB size limit"));
    }

    #[tokio::test]
    async fn fal_success_body_rejects_oversized_chunked_response() {
        let (response, release, server) =
            held_open_chunked_response(FAL_RESPONSE_LIMIT_BYTES + 1).await;

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            read_fal_success_body(response),
        )
        .await;
        finish_held_open_response(release, server).await;
        let error = result
            .expect("fal.ai success reader waited for EOF after crossing its limit")
            .unwrap_err();

        assert!(error.to_string().contains("1 MiB size limit"));
    }

    #[tokio::test]
    async fn fal_error_body_is_bounded_during_streaming() {
        let (response, release, server) =
            held_open_chunked_response(FAL_ERROR_LIMIT_BYTES + 1).await;

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            read_fal_error_text(response),
        )
        .await;
        finish_held_open_response(release, server).await;
        let body = result
            .expect("fal.ai error reader waited for EOF after crossing its limit")
            .unwrap();

        assert_eq!(body.len(), FAL_ERROR_LIMIT_BYTES);
        assert!(body.bytes().all(|byte| byte == b'x'));
    }

    #[tokio::test]
    async fn fal_client_does_not_follow_redirects() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = zeroclaw_spawn::spawn!(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/unchecked\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
        });

        let response = ImageGenTool::http_client()
            .unwrap()
            .get(format!("http://{addr}/start"))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        server.await.unwrap();
    }

    #[test]
    fn tool_name() {
        let tool = test_tool();
        assert_eq!(tool.name(), "image_gen");
    }

    #[test]
    fn tool_description_is_nonempty() {
        let tool = test_tool();
        assert!(!tool.description().is_empty());
        assert!(tool.description().contains("image"));
    }

    #[test]
    fn tool_schema_has_required_prompt() {
        let tool = test_tool();
        let schema = tool.parameters_schema();
        assert_eq!(schema["required"], json!(["prompt"]));
        assert!(schema["properties"]["prompt"].is_object());
    }

    #[test]
    fn tool_schema_has_optional_params() {
        let tool = test_tool();
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["filename"].is_object());
        assert!(schema["properties"]["size"].is_object());
        assert!(schema["properties"]["model"].is_object());
    }

    #[test]
    fn tool_spec_roundtrip() {
        let tool = test_tool();
        let spec = tool.spec();
        assert_eq!(spec.name, "image_gen");
        assert!(spec.parameters.is_object());
    }

    #[tokio::test]
    async fn missing_prompt_returns_error() {
        let tool = test_tool();
        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("prompt"));
    }

    #[tokio::test]
    async fn empty_prompt_returns_error() {
        let tool = test_tool();
        let result = tool.execute(json!({"prompt": "   "})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("prompt"));
    }

    #[tokio::test]
    async fn missing_api_key_returns_error() {
        // Temporarily ensure the env var is unset.
        let original = std::env::var("FAL_API_KEY_TEST_IMAGE_GEN").ok();
        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::remove_var("FAL_API_KEY_TEST_IMAGE_GEN") };

        let tool = ImageGenTool::new(
            test_security(),
            std::env::temp_dir(),
            "fal-ai/flux/schnell".into(),
            "FAL_API_KEY_TEST_IMAGE_GEN".into(),
            Vec::new(),
        )
        .unwrap();
        let result = tool
            .execute(json!({"prompt": "a sunset over the ocean"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap()
                .contains("FAL_API_KEY_TEST_IMAGE_GEN")
        );

        // Restore if it was set.
        if let Some(val) = original {
            // SAFETY: test-only, single-threaded test runner.
            unsafe { std::env::set_var("FAL_API_KEY_TEST_IMAGE_GEN", val) };
        }
    }

    #[tokio::test]
    async fn invalid_size_returns_error() {
        // Set a dummy key so we get past the key check.
        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::set_var("FAL_API_KEY_TEST_SIZE", "dummy_key") };

        let tool = ImageGenTool::new(
            test_security(),
            std::env::temp_dir(),
            "fal-ai/flux/schnell".into(),
            "FAL_API_KEY_TEST_SIZE".into(),
            Vec::new(),
        )
        .unwrap();
        let result = tool
            .execute(json!({"prompt": "test", "size": "invalid_size"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("Invalid size"));

        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::remove_var("FAL_API_KEY_TEST_SIZE") };
    }

    #[tokio::test]
    async fn read_only_autonomy_blocks_execution() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });
        let tool = ImageGenTool::new(
            security,
            std::env::temp_dir(),
            "fal-ai/flux/schnell".into(),
            "FAL_API_KEY".into(),
            Vec::new(),
        )
        .unwrap();
        let result = tool.execute(json!({"prompt": "test image"})).await.unwrap();
        assert!(!result.success);
        let err = result.error.as_deref().unwrap();
        assert!(
            err.contains("read-only") || err.contains("image_gen"),
            "expected read-only or image_gen in error, got: {err}"
        );
    }

    #[tokio::test]
    async fn invalid_model_with_traversal_returns_error() {
        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::set_var("FAL_API_KEY_TEST_MODEL", "dummy_key") };

        let tool = ImageGenTool::new(
            test_security(),
            std::env::temp_dir(),
            "fal-ai/flux/schnell".into(),
            "FAL_API_KEY_TEST_MODEL".into(),
            Vec::new(),
        )
        .unwrap();
        let result = tool
            .execute(json!({"prompt": "test", "model": "../../evil-endpoint"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap()
                .contains("Invalid model identifier")
        );

        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::remove_var("FAL_API_KEY_TEST_MODEL") };
    }

    #[test]
    fn read_api_key_missing() {
        let result = ImageGenTool::read_api_key("DEFINITELY_NOT_SET_ZC_TEST_12345");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("DEFINITELY_NOT_SET_ZC_TEST_12345")
        );
    }

    #[test]
    fn filename_traversal_is_sanitized() {
        // Verify that path traversal in filenames is stripped to just the final component.
        let sanitized = PathBuf::from("../../etc/passwd").file_name().map_or_else(
            || "generated_image".to_string(),
            |n| n.to_string_lossy().to_string(),
        );
        assert_eq!(sanitized, "passwd");

        // ".." alone has no file_name, falls back to default.
        let sanitized = PathBuf::from("..").file_name().map_or_else(
            || "generated_image".to_string(),
            |n| n.to_string_lossy().to_string(),
        );
        assert_eq!(sanitized, "generated_image");
    }

    #[test]
    fn resolve_image_filename_default_is_non_clobbering_and_unique() {
        // Exercises the PRODUCTION filename-selection helper an omitted
        // filename must yield a unique timestamped name, never the bare
        // `generated_image` that would clobber prior generations, and two
        // default calls must differ. Fails if the code reverts to a fixed name.
        let a = resolve_image_filename(None, 1_000);
        let b = resolve_image_filename(None, 2_000);
        assert_eq!(a, "generated_image_1000");
        assert_ne!(
            a, "generated_image",
            "default must not clobber the bare name"
        );
        assert_ne!(a, b, "successive default names must differ");
        // An explicit filename is used verbatim, with path components stripped.
        assert_eq!(resolve_image_filename(Some("my_pic"), 1_000), "my_pic");
        assert_eq!(
            resolve_image_filename(Some("../../etc/passwd"), 1_000),
            "passwd"
        );
        // Blank/whitespace filename falls back to the timestamped default.
        assert_eq!(
            resolve_image_filename(Some("   "), 1_000),
            "generated_image_1000"
        );
    }

    #[test]
    fn ephemeral_warning_keeps_the_declared_attachment() {
        let marker = RenderedMarker {
            target: "/ws/images/generated_image_42.png".into(),
            kind: MarkerKind::Image,
        };
        let generated = ToolResult {
            success: true,
            output: "File: /ws/images/generated_image_42.png".into(),
            error: None,
        }
        .with_attachment(marker.clone());

        let warned = warn_if_ephemeral(false, generated.clone());
        assert!(
            warned.output.contains("EPHEMERAL WORKSPACE"),
            "ephemeral warning must be present, got: {}",
            warned.output
        );
        assert!(
            warned
                .output
                .contains("File: /ws/images/generated_image_42.png")
        );
        assert_eq!(warned.output.attachments(), std::slice::from_ref(&marker));

        let persistent = warn_if_ephemeral(true, generated);
        assert!(!persistent.output.contains("EPHEMERAL WORKSPACE"));
        assert_eq!(
            persistent.output.attachments(),
            std::slice::from_ref(&marker)
        );
    }

    #[test]
    fn image_output_carries_file_line_and_no_inline_marker() {
        let path = "/ws/images/generated_image_42.png";
        let out = format_image_tool_output(path, 12, "fal-ai/flux", "a cat");
        assert!(
            out.contains(&format!("File: {path}")),
            "output must carry a durable File: line: {out}"
        );
        assert!(
            !out.contains("[IMAGE:"),
            "the body must not carry marker syntax: {out}"
        );
    }

    #[test]
    fn read_api_key_present() {
        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::set_var("ZC_IMAGE_GEN_TEST_KEY", "test_value_123") };
        let result = ImageGenTool::read_api_key("ZC_IMAGE_GEN_TEST_KEY");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test_value_123");
        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::remove_var("ZC_IMAGE_GEN_TEST_KEY") };
    }
}
