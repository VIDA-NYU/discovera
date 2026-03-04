"""
Very simple file API HTTP server.

API:
- POST /upload
  - multipart/form-data with field name "file"
  - or raw body with ?filename=... or X-Filename header
- GET /download?filename=...
- PUT /update?filename=...
  - multipart/form-data with field name "file", or raw bytes
- DELETE /delete?filename=...

Files are saved to UPLOAD_SERVER_OUTPUT_DIR
(default: /Users/yifanwu/Desktop/VIDA/ARPA-H/discovera/output)
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

OUTPUT_DIR = Path(
    os.getenv("UPLOAD_SERVER_OUTPUT_DIR", "/Users/yifanwu/Desktop/VIDA/ARPA-H/discovera/output")
)
HOST = os.getenv("UPLOAD_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("UPLOAD_SERVER_PORT", "8001"))
MAX_UPLOAD_BYTES = int(os.getenv("UPLOAD_SERVER_MAX_BYTES", str(100 * 1024 * 1024)))


def _safe_filename(name: str | None) -> str:
    if not name:
        return f"upload_{int(time.time())}.bin"
    basename = Path(name).name.strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", basename)
    return sanitized or f"upload_{int(time.time())}.bin"


def _parse_multipart_file(content_type: str, body: bytes) -> tuple[str | None, bytes]:
    # Build a synthetic MIME message so `email` can parse multipart/form-data.
    payload = (
        b"MIME-Version: 1.0\r\n"
        + f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        + body
    )
    msg = BytesParser(policy=default).parsebytes(payload)
    if not msg.is_multipart():
        return None, b""

    for part in msg.iter_parts():
        if part.get_param("name", header="content-disposition") != "file":
            continue
        filename = part.get_filename()
        data = part.get_payload(decode=True) or b""
        return filename, data

    return None, b""


def _resolve_filename_from_query(parsed_path: str) -> str | None:
    query = parse_qs(urlparse(parsed_path).query)
    raw = query.get("filename", [None])[0]
    if raw is None:
        return None
    return _safe_filename(raw)


def _resolve_path_by_name(filename: str) -> Path:
    return OUTPUT_DIR / _safe_filename(filename)


class UploadHandler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            self._json(
                HTTPStatus.LENGTH_REQUIRED,
                {"ok": False, "error": "Content-Length header is required"},
            )
            return

        try:
            content_length = int(raw_len)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid length"})
            return

        if content_length <= 0:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Empty upload"})
            return

        if content_length > MAX_UPLOAD_BYTES:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"ok": False, "error": "Upload too large"},
            )
            return

        content_type = self.headers.get("Content-Type", "")
        filename = None

        body = self.rfile.read(content_length)
        data = b""

        if content_type.startswith("multipart/form-data"):
            filename, data = _parse_multipart_file(content_type, body)
            if not data:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "Use multipart field named 'file'"},
                )
                return
        else:
            query = parse_qs(parsed.query)
            filename = query.get("filename", [None])[0] or self.headers.get("X-Filename")
            data = body

        if not data:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "No file bytes found"})
            return

        safe_name = _safe_filename(filename)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Keep only the newest duplicate: same filename overwrites older file.
        target = OUTPUT_DIR / safe_name
        target.write_bytes(data)

        self._json(
            HTTPStatus.CREATED,
            {
                "ok": True,
                "filename": target.name,
                "saved_to": str(target),
                "size_bytes": len(data),
            },
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/download":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        safe_name = _resolve_filename_from_query(self.path)
        if not safe_name:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Missing query parameter: filename"},
            )
            return

        target = _resolve_path_by_name(safe_name)
        if not target.exists() or not target.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "File not found"})
            return

        data = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(data)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/update":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            self._json(
                HTTPStatus.LENGTH_REQUIRED,
                {"ok": False, "error": "Content-Length header is required"},
            )
            return

        try:
            content_length = int(raw_len)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid length"})
            return

        if content_length <= 0:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Empty upload"})
            return

        if content_length > MAX_UPLOAD_BYTES:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"ok": False, "error": "Upload too large"},
            )
            return

        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(content_length)
        provided_name = _resolve_filename_from_query(self.path)

        if content_type.startswith("multipart/form-data"):
            part_name, data = _parse_multipart_file(content_type, body)
            if not data:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "Use multipart field named 'file'"},
                )
                return
            safe_name = provided_name or _safe_filename(part_name)
        else:
            data = body
            safe_name = provided_name

        if not safe_name:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {
                    "ok": False,
                    "error": "Missing filename. Use ?filename=... or multipart file name",
                },
            )
            return

        target = _resolve_path_by_name(safe_name)
        if not target.exists() or not target.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "File not found"})
            return

        target.write_bytes(data)
        self._json(
            HTTPStatus.OK,
            {
                "ok": True,
                "filename": target.name,
                "saved_to": str(target),
                "size_bytes": len(data),
            },
        )

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/delete":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        safe_name = _resolve_filename_from_query(self.path)
        if not safe_name:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Missing query parameter: filename"},
            )
            return

        target = _resolve_path_by_name(safe_name)
        if not target.exists() or not target.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "File not found"})
            return

        target.unlink()
        self._json(HTTPStatus.OK, {"ok": True, "deleted": target.name})

    def log_message(self, fmt: str, *args) -> None:
        # Keep server output clean and concise.
        print(f"[upload-server] {self.address_string()} - {fmt % args}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), UploadHandler)
    print(f"File API server listening on http://{HOST}:{PORT}")
    print("Endpoints: POST /upload, GET /download, PUT /update, DELETE /delete")
    print(f"Saving files to: {OUTPUT_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
