# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Tuple, Optional


class InMemoryHTTPServer:
    """
    Serve in-memory binary content over HTTP from a mapping of path -> (bytes, content_type).

    - Start/stop the server with start()/stop().
    - Add/update content with add() or set().
    - Remove content with remove().
    - Check if a path exists with has().
    - List available paths with paths().

    Example:
        srv = InMemoryHTTPServer(host="127.0.0.1", port=8001)
        srv.start()
        srv.add("/img/current.png", png_bytes, "image/png")
        url = srv.url("/img/current.png")  # "http://127.0.0.1:8001/img/current.png"
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8001):
        self.host = host
        self.port = port
        self._content: Dict[str, Tuple[bytes, str]] = {}
        self._lock = threading.RLock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Build a request handler class bound to this instance
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def _set_cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")

            def do_OPTIONS(self):
                self.send_response(204)
                self._set_cors()
                self.end_headers()

            def do_GET(self):
                # Normalize path (no query for path lookup)
                path = self.path.split("?", 1)[0]
                with outer._lock:
                    hit = outer._content.get(path)

                if hit is None:
                    self.send_response(404)
                    self.end_headers()
                    return

                data, ctype = hit
                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                # Basic caching headers; adjust as needed
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                # Silence default logging; customize if needed
                return

        self._handler_cls = _Handler

    # --- Server lifecycle ---

    def start(self):
        if self._httpd is not None:
            return  # already running
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler_cls)

        def _serve():
            try:
                self._httpd.serve_forever()
            except Exception:
                pass

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def stop(self):
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None

    # --- Content management ---

    def add(self, path: str, data: bytes, content_type: str):
        """
        Add or replace a path with bytes and content type.
        Path must start with '/'.
        """
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes or bytearray")
        with self._lock:
            self._content[path] = (bytes(data), content_type)

    # Alias for add
    def set(self, path: str, data: bytes, content_type: str):
        self.add(path, data, content_type)

    def remove(self, path: str):
        with self._lock:
            self._content.pop(path, None)

    def clear(self):
        with self._lock:
            self._content.clear()

    def has(self, path: str) -> bool:
        with self._lock:
            return path in self._content

    def paths(self):
        with self._lock:
            return list(self._content.keys())

    def url(self, path: str) -> str:
        """
        Build a full http:// URL for a stored path.
        """
        if not path.startswith("/"):
            path = "/" + path
        return f"http://{self.host}:{self.port}{path}"
