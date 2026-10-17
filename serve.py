#!/usr/bin/env python3
"""Static server for brainview (web/).  No-cache so edits show on reload.

The data files are plain *.bin.gz files that the viewer inflates itself, so this is an ordinary
static server — the same files work unchanged on GitHub Pages.

Usage: python3 serve.py [port]   (default 8051, bound to 127.0.0.1; exposed via `tailscale serve`)
"""
import functools
import http.server
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8051
    handler = functools.partial(Handler, directory=ROOT)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
