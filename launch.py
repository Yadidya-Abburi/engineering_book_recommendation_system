#!/usr/bin/env python3
"""
Bookify Launcher — run this to open Bookify in your browser.
Usage: python launch.py
"""
import http.server, webbrowser, threading, os, time

PORT = 8765
ROOT = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)
    def log_message(self, *_):
        pass  # suppress request logs

def open_browser():
    time.sleep(0.8)
    webbrowser.open(f"http://localhost:{PORT}/app/index.html")

print(f"Starting Bookify on http://localhost:{PORT}/app/index.html")
print("Press Ctrl+C to stop.\n")

threading.Thread(target=open_browser, daemon=True).start()

with http.server.HTTPServer(("", PORT), Handler) as srv:
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nBookify stopped.")
