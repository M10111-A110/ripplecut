"""Local demo UI server (master spec §67-§72, §104). Standard library only; works offline.

Endpoints (JSON):
  GET  /                 the single-page UI (ui/static/index.html; no CDN, no external assets)
  GET  /api/model        topology, rules, criticality, actions, solvers, policy, scenarios
  POST /api/parse        {"text": ...}                         -> parse result (structured incident or ambiguity)
  POST /api/run          {"scenario_id"| "incident" | "text", "fault": {"mode": ...}?} -> run report
  POST /api/approve      {"run_id": ...}                       -> SIMULATED approval record (nothing executed)

The UI is a view: every value it shows comes from the run report produced by
the deterministic pipeline. It never computes cascades or plans itself.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from ..errors import ErrorCode, RippleCutError
from ..pipeline import FaultSpec, RippleCutApp

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 64 * 1024
MAX_RUNS = 200


class DemoState:
    def __init__(self, app: RippleCutApp, allow_fault_injection: bool = True,
                 audit_dir: Optional[Path] = None) -> None:
        self.app = app
        self.allow_fault_injection = allow_fault_injection
        self.audit_dir = audit_dir or Path("audit")
        self.runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.lock = threading.Lock()

    def remember(self, report: Dict[str, Any]) -> None:
        with self.lock:
            self.runs[report["run_id"]] = report
            while len(self.runs) > MAX_RUNS:
                self.runs.popitem(last=False)

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.runs.get(run_id)


def handle_api(state: DemoState, method: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    app = state.app
    if method == "GET" and path == "/api/model":
        return app.model_view()
    if method == "POST" and path == "/api/parse":
        text = body.get("text")
        if not isinstance(text, str):
            raise RippleCutError("'text' must be a string", ErrorCode.CONFIG_ERROR)
        return app.parse_text(text).to_dict()
    if method == "POST" and path == "/api/run":
        fault = None
        if body.get("fault"):
            if not state.allow_fault_injection:
                raise RippleCutError("fault injection is disabled in this environment", ErrorCode.CONFIG_ERROR)
            f = body["fault"]
            if not isinstance(f, dict):
                raise RippleCutError("'fault' must be an object", ErrorCode.CONFIG_ERROR)
            fault = FaultSpec(mode=str(f.get("mode")), solver=f.get("solver"),
                              timeout_seconds=float(f.get("timeout_seconds", 1.5)))
        if body.get("scenario_id"):
            sid = str(body["scenario_id"])
            if not any(s["id"] == sid for s in app.scenarios):
                valid_ids = [s["id"] for s in app.scenarios]
                raise RippleCutError(f"unknown scenario_id '{sid}'; valid scenarios: {valid_ids}", ErrorCode.CONFIG_ERROR)
            report = app.run_scenario(sid, fault=fault)
        elif body.get("incident") is not None:
            report = app.run_incident(body["incident"], fault=fault)
        elif isinstance(body.get("text"), str):
            report = app.run_text(body["text"], fault=fault)
        else:
            raise RippleCutError("provide 'scenario_id', 'incident' or 'text'", ErrorCode.CONFIG_ERROR)
        state.remember(report)
        return report
    if method == "POST" and path == "/api/approve":
        report = state.get(str(body.get("run_id", "")))
        if report is None:
            raise RippleCutError("unknown run_id; run the planner first", ErrorCode.CONFIG_ERROR)
        if report.get("status") != "SUCCESS":
            raise RippleCutError("there is no recommended plan to approve for this run", ErrorCode.CONFIG_ERROR)

        res = report.get("result", {}) or {}
        obj = res.get("objective")
        scenario_id = (report.get("incident", {}) or {}).get("scenario_id") or report.get("scenario_id") or report.get("problem_id", "")
        solver_used = res.get("selected_solver") or (report.get("planner", {}) or {}).get("selected_solver") or "unknown"
        optimality = res.get("optimality_status", "UNKNOWN")
        val_status = res.get("validation_status") or ("VALID" if (report.get("validation", {}) or {}).get("valid") else "INVALID")

        record = {
            "run_id": report["run_id"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario_id": scenario_id,
            "incident_id": report.get("problem_id", ""),
            "plan": res.get("plan"),
            "objective": obj,
            "solver": solver_used,
            "optimality_status": optimality,
            "validation_status": val_status,
            "status": "APPROVED_SIMULATED",
            "approval_status": "APPROVED_SIMULATED",
            "executed": False,
            "approved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Simulated approval for the demo. RippleCut executed nothing; applying the plan to a real "
                    "system is a separate, human-operated step outside this MVP."
        }
        # Persist audit record. Do not silently swallow persistence errors.
        audit_dir = getattr(state, "audit_dir", Path("audit"))
        try:
            audit_dir.mkdir(parents=True, exist_ok=True)
            with (audit_dir / "approvals.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            raise RippleCutError(f"Audit log persistence failed: {exc}; approval cannot be recorded",
                                 ErrorCode.INTERNAL_ERROR) from exc

        report["approval"] = {**report.get("approval", {}), **record}
        return record
    raise RippleCutError(f"no route {method} {path}", ErrorCode.CONFIG_ERROR, {"http_status": 404})


def make_handler(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RippleCutDemo/1.0"
        timeout = 10.0

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet default access log
            pass

        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, code: int, data: Any) -> None:
            self._send(code, json.dumps(data, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            if method == "GET" and path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            body: Dict[str, Any] = {}
            if method == "POST":
                raw_len = (self.headers.get("Content-Length") or "").strip()
                try:
                    length = int(raw_len) if raw_len else 0
                except ValueError:
                    self._json(400, {"code": "CONFIG_ERROR", "message": "invalid Content-Length"})
                    return
                if length < 0:
                    self._json(400, {"code": "CONFIG_ERROR", "message": "negative Content-Length"})
                    return
                if length > MAX_BODY:
                    self._json(413, {"code": "CONFIG_ERROR", "message": "request body too large"})
                    return
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    self._json(400, {"code": "CONFIG_ERROR", "message": "request body must be a JSON object"})
                    return
            try:
                self._json(200, handle_api(state, method, path, body))
            except RippleCutError as exc:
                self._json(exc.details.get("http_status", 400), exc.to_dict())
            except Exception as exc:  # noqa: BLE001 - surface, never hide, unexpected failures
                self._json(500, {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    return Handler


def serve(app: RippleCutApp, host: str = "127.0.0.1", port: int = 8765,
          allow_fault_injection: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(DemoState(app, allow_fault_injection=allow_fault_injection)))
    print(f"RippleCut demo UI (offline) at http://{host}:{port}/  - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
