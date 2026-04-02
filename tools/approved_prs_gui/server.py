#!/usr/bin/env python3
#####################################################################################
# Small local server for the approved-PRs viewer. Fetches from the GitHub Search API
# (avoids browser CORS). Optional: set GITHUB_TOKEN for a higher rate limit.
#
# Usage: python3 tools/approved_prs_gui/server.py
# Then open http://127.0.0.1:8765/
#####################################################################################
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = "ROCm/AMDMIGraphX"
HOST = os.environ.get("APPROVED_PRS_GUI_HOST", "127.0.0.1")
PORT = int(os.environ.get("APPROVED_PRS_GUI_PORT", "8765"))
ROOT = Path(__file__).resolve().parent

GITHUB_ACCEPT = "application/vnd.github+json"
USER_AGENT = "AMDMIGraphX-approved-prs-gui/1.0"


def _github_headers() -> dict[str, str]:
    h = {
        "Accept": GITHUB_ACCEPT,
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token.strip()}"
    return h


def _fetch_json(url: str) -> tuple[dict | list, dict]:
    req = urllib.request.Request(url, headers=_github_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode()
        data = json.loads(body)
        remaining = resp.headers.get("X-RateLimit-Remaining", "")
        reset = resp.headers.get("X-RateLimit-Reset", "")
        meta = {"remaining": remaining, "reset": reset}
        return data, meta


def search_approved_prs(state: str) -> tuple[list[dict], dict | None]:
    """
    state: 'open' | 'closed' | 'all'
    Returns normalized PR rows and optional error dict.
    """
    q_parts = [f"repo:{REPO}", "is:pr", "review:approved"]
    if state == "open":
        q_parts.append("is:open")
    elif state == "closed":
        q_parts.append("is:closed")
    elif state != "all":
        return [], {"error": "invalid_state", "message": "state must be open, closed, or all"}

    per_page = 100
    page = 1
    items: list[dict] = []
    rate_meta: dict = {}
    total = 0

    while True:
        params = urllib.parse.urlencode(
            {"q": " ".join(q_parts), "per_page": per_page, "page": page}
        )
        url = f"https://api.github.com/search/issues?{params}"
        try:
            data, meta = _fetch_json(url)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode()
                detail = json.loads(err_body)
            except Exception:
                detail = {"message": str(e)}
            return [], {
                "error": "http_error",
                "status": e.code,
                "message": detail.get("message", str(e)),
            }
        except Exception as e:
            return [], {"error": "request_failed", "message": str(e)}

        rate_meta.update(meta)
        batch = data.get("items") or []
        for it in batch:
            pr = it.get("pull_request") or {}
            items.append(
                {
                    "number": it.get("number"),
                    "title": it.get("title"),
                    "state": it.get("state"),
                    "html_url": it.get("html_url"),
                    "updated_at": it.get("updated_at"),
                    "created_at": it.get("created_at"),
                    "user": (it.get("user") or {}).get("login"),
                    "draft": bool(pr.get("draft", False)),
                }
            )

        total = int(data.get("total_count") or 0)
        if (
            len(batch) < per_page
            or page * per_page >= total
            or page * per_page >= 1000
        ):
            break
        page += 1

    meta_out = {
        "rate_limit": rate_meta,
        "total_reported": total,
        "truncated": total > len(items),
    }
    return items, meta_out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/prs":
            qs = urllib.parse.parse_qs(parsed.query or "")
            state = (qs.get("state") or ["open"])[0].lower()
            rows, err = search_approved_prs(state)
            payload: dict = {"repository": REPO, "state_filter": state, "pull_requests": rows}
            if err and "error" in err:
                payload["error"] = err
            elif err:
                payload["meta"] = err
            self._send_json(payload)
            return

        if parsed.path == "/" or parsed.path == "":
            path = ROOT / "index.html"
        else:
            rel = parsed.path.lstrip("/")
            if ".." in rel or rel.startswith("/"):
                self.send_error(404)
                return
            path = ROOT / rel

        if not path.is_file() or not str(path.resolve()).startswith(str(ROOT.resolve())):
            self.send_error(404)
            return

        ctype = "text/html" if path.suffix == ".html" else "application/octet-stream"
        if path.suffix == ".css":
            ctype = "text/css"
        elif path.suffix == ".js":
            ctype = "application/javascript"

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: dict) -> None:
        raw = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    httpd = HTTPServer((HOST, PORT), Handler)
    print(f"Approved PRs GUI: http://{HOST}:{PORT}/")
    print(f"Repository: {REPO} (set GITHUB_TOKEN for higher API limits)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
