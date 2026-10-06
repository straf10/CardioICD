"""Start the CardioICD demo with the models already loading.

    python app/run_demo.py            # then open http://localhost:8501
    python app/run_demo.py --port 8502

``streamlit run app/streamlit_app.py`` works too, but Streamlit only runs the page script once a
browser connects, so model loading would start on the first visit. This launcher starts loading
immediately, in the same process, and the page picks up the same models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import common as cm  # noqa: E402

cm.prefer_offline_hub()  # before anything imports huggingface_hub
import loader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    missing = cm.missing_artifacts()
    if missing:
        print("Can't start loading, these files are missing:\n  " + "\n  ".join(missing), file=sys.stderr)
    else:
        loader.start()
        print("Loading the models in the background...")

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(APP / "streamlit_app.py"),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.no_browser else "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
