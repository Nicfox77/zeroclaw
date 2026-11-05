#!/usr/bin/env python3
"""Transport and gateway-boundary tests; no external services or credentials."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import config_control as bridge


class Fixture:
    def __init__(self):
        self.calls = []
        self.token = "fixture-private-token"
        self.fail_status = False
        self.reject_patch = False
        self.running = []
        self.patches = []
        self.redirect = False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.handle_request()

            def do_PATCH(self):
                self.handle_request()

            def do_POST(self):
                self.handle_request()

            def handle_request(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else None
                fixture.calls.append((self.command, self.path, body))
                if self.headers.get("Authorization") != "Bearer " + fixture.token:
                    self.reply(401, {"error": "unauthorized"})
                elif fixture.redirect:
                    self.send_response(302)
                    self.send_header("Location", "/redirect-destination")
                    self.end_headers()
                elif self.path == "/api/config" and self.command == "PATCH":
                    if fixture.reject_patch:
                        self.reply(422, {"code": "validation_failed", "message": fixture.token})
                    else:
                        fixture.patches.append(body)
                        self.reply(200, {"value": fixture.token})
                elif self.path == "/api/config/reload-status":
                    self.reply(503 if fixture.fail_status else 200, {"pending_reload": True})
                elif self.path == "/api/sessions/running":
                    self.reply(200, fixture.running.pop(0) if fixture.running else {"sessions": []})
                elif self.path == "/admin/reload":
                    self.reply(200, {"success": True})
                elif self.path.startswith("/api/config/map-key?"):
                    self.reply(200, {"created": True, "value": fixture.token})
                elif self.path.startswith("/api/config/prop?"):
                    self.reply(200, {"path": "fixture.api_key", "populated": True})
                elif self.path.startswith("/api/config/list?"):
                    self.reply(200, {"entries": [{"path": f"fixture.{i}", "value": str(i)} for i in range(5)]})
                else:
                    self.reply(404, {"error": "not_found"})

            def reply(self, status, body):
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class ConfigControlTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.fixture = Fixture()
        self.token_file = self.root / "token"
        self.token_file.write_text(self.fixture.token)
        self.token_file.chmod(0o600)
        self.gateway = bridge.Gateway(self.fixture.url, self.token_file)
        self.control = bridge.Control(self.gateway, self.root / "jobs")

    def tearDown(self):
        self.fixture.close()
        self.directory.cleanup()

    def test_atomic_patch_is_forwarded_once_without_echoing_values(self):
        operations = [{"op": "test", "path": "agent.limit", "value": 50},
                      {"op": "replace", "path": "agent.limit", "value": 200}]
        result = self.control.call("patch", {"operations": operations})
        self.assertEqual(self.fixture.patches, [operations])
        self.assertTrue(result["saved"])
        self.assertNotIn(self.fixture.token, json.dumps(result))

    def test_rejected_patch_returns_only_safe_metadata(self):
        self.fixture.reject_patch = True
        with self.assertRaises(bridge.BridgeError) as caught:
            self.control.call("set", {"path": "fixture.limit", "value": 200})
        self.assertEqual(caught.exception.details["status"], 422)
        self.assertEqual(caught.exception.details["code"], "validation_failed")
        self.assertNotIn(self.fixture.token, json.dumps(caught.exception.details))
        self.assertEqual(self.fixture.patches, [])

    def test_status_failure_cannot_hide_a_successful_write(self):
        self.fixture.fail_status = True
        result = self.control.call("set", {"path": "fixture.limit", "value": 200})
        self.assertTrue(result["saved"])
        self.assertTrue(result["application_status_unavailable"])
        self.assertEqual(len(self.fixture.patches), 1)

    def test_secrets_use_the_masked_gateway_property_surface(self):
        result = self.control.call("get", {"path": "fixture.api_key"})
        self.assertEqual(result, {"path": "fixture.api_key", "populated": True})
        self.assertNotIn(self.fixture.token, json.dumps(result))

    def test_pagination_and_invalid_batches(self):
        result = self.control.call("list", {"prefix": "fixture", "offset": 1, "limit": 2})
        self.assertEqual([entry["value"] for entry in result["entries"]], ["1", "2"])
        self.assertEqual(result["next_offset"], 3)
        before = len(self.fixture.calls)
        for arguments in ({"operations": []}, {"operations": [{"op": "copy", "path": "x"}]},
                          {"operations": [{"op": "add", "path": "x"}]},
                          {"operations": [{"op": "add", "path": "x", "value": 1, "extra": True}]}):
            with self.assertRaises(bridge.BridgeError):
                self.control.call("patch", arguments)
        self.assertEqual(len(self.fixture.calls), before)

    def test_alias_creation_uses_canonical_gateway_api(self):
        result = self.control.call("create_alias", {"path": "mcp.servers", "key": "fixture"})
        self.assertTrue(result["created"])
        self.assertEqual(self.fixture.calls[-1][1], "/api/config/map-key?path=mcp.servers&key=fixture")
        self.assertNotIn(self.fixture.token, json.dumps(result))

    def test_token_rotation_and_file_protection(self):
        self.control.call("get", {"path": "fixture.api_key"})
        self.fixture.token = "rotated-fixture-token"
        self.token_file.write_text(self.fixture.token)
        self.control.call("get", {"path": "fixture.api_key"})
        self.token_file.chmod(0o644)
        with self.assertRaises(bridge.BridgeError):
            self.control.call("get", {"path": "fixture.api_key"})
        self.token_file.chmod(0o600)
        link = self.root / "token-link"
        link.symlink_to(self.token_file)
        with self.assertRaises(OSError):
            bridge.Gateway(self.fixture.url, link).token()

    def test_redirects_and_non_loopback_targets_are_refused(self):
        for url in ("https://127.0.0.1:1234", "http://example.invalid:1234", "http://127.0.0.1:1234/?x=1",
                    "http://user:password@127.0.0.1:1234", "http://0.0.0.0:1234"):
            with self.assertRaises(bridge.BridgeError):
                bridge.Gateway(url, self.token_file)
        self.fixture.redirect = True
        with self.assertRaises(bridge.BridgeError):
            self.control.call("get", {"path": "fixture.api_key"})
        self.assertEqual(len(self.fixture.calls), 1)

    def test_uncertain_mutations_are_never_retried(self):
        with patch.object(self.gateway.http, "open", side_effect=TimeoutError) as send:
            with self.assertRaises(bridge.BridgeError) as caught:
                self.control.call("set", {"path": "fixture.limit", "value": 200})
        self.assertEqual(send.call_count, 1)
        self.assertTrue(caught.exception.details["outcome_unknown"])

    def test_reload_waits_for_idle_and_records_only_request_acceptance(self):
        job = self.root / "job.json"
        bridge.write_job(job, {"state": "waiting_for_idle"})
        self.fixture.running = [{"sessions": [{"session_id": "fixture"}]}, {"sessions": []}, {"sessions": []}]
        with patch.object(bridge.time, "sleep"):
            bridge.run_reload(self.gateway, job, idle_seconds=0)
        self.assertEqual(self.fixture.calls[-1][1], "/admin/reload")
        self.assertEqual(json.loads(job.read_text())["state"], "reload_requested")

    def test_reload_fails_closed_without_native_idle_evidence(self):
        job = self.root / "job.json"
        bridge.write_job(job, {"state": "waiting_for_idle"})
        self.fixture.running = [{"sessions": [], "message": "Session persistence is disabled"}]
        bridge.run_reload(self.gateway, job, idle_seconds=0)
        self.assertEqual(json.loads(job.read_text())["state"], "failed")
        self.assertFalse(any(call[1] == "/admin/reload" for call in self.fixture.calls))

    def test_reload_returns_before_worker_and_never_passes_the_token(self):
        with patch.object(bridge.subprocess, "Popen") as spawn:
            result = self.control.call("reload", {})
        self.assertEqual(result["state"], "waiting_for_idle")
        self.assertNotIn(self.fixture.token, json.dumps(spawn.call_args.args))
        self.assertEqual(list((self.root / "jobs").glob("*.json"))[0].stat().st_mode & 0o777, 0o600)

    def test_real_stdio_process_discovers_tools_and_reaches_gateway(self):
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "set", "arguments": {"path": "fixture.limit", "value": 200}}},
        ]
        result = subprocess.run([sys.executable, str(Path(bridge.__file__)), "--gateway-url", self.fixture.url,
                                 "--token-file", str(self.token_file), "--state-dir", str(self.root / "jobs")],
                                input="".join(json.dumps(value) + "\n" for value in requests),
                                capture_output=True, text=True, timeout=10, check=True)
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([reply["id"] for reply in replies], [1, 2, 3])
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(len(replies[1]["result"]["tools"]), 7)
        self.assertFalse(replies[2]["result"]["isError"])
        self.assertEqual(len(self.fixture.patches), 1)
        self.assertNotIn(self.fixture.token, result.stdout + result.stderr)

    def test_oversized_or_invalid_requests_cannot_execute_a_tail(self):
        output = io.StringIO()
        bridge.serve(self.control, io.BytesIO(b"x" * (bridge.MAX_MESSAGE + 1) + b"\n"), output)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(self.fixture.calls, [])


if __name__ == "__main__":
    unittest.main()
