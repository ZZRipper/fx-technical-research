"""Serve the packaged dashboard locally. Python standard library only."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parent / "research"
PAGE = "FX_PnL_Explorer_Simple.html"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="0 selects an available port")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check packaged entry points without serving")
    args = parser.parse_args()
    for relative in (PAGE, "build_simple_dashboard.py", "build_simple_data.py", "reproduced/simple_interest/explorer_payload.json.gz"):
        if not (ROOT / relative).is_file():
            raise SystemExit("Missing file: " + relative + ". Extract the complete ZIP first.")
    if args.check:
        print("Package entry points found.")
        return
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as exc:
        raise SystemExit(str(exc) + ". Retry without --port to use an available port.")
    with server:
        url = "http://127.0.0.1:" + str(server.server_port) + "/" + PAGE + "?view=timeline#comparison"
        print("Dashboard: " + url, flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop. No internet connection is required.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")

if __name__ == "__main__":
    main()
