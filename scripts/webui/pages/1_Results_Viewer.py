from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "scripts").is_dir():
            return ancestor
    return here.parents[3]


def _import_app_and_run() -> None:
    root = _project_root()
    app_path = root / "scripts" / "webui" / "app.py"
    if not app_path.exists():
        import streamlit as st
        st.error(f"Missing module: {app_path}")
        return
    sys.path.append(str(app_path.parent))
    app_module = importlib.import_module("app")
    if hasattr(app_module, "main"):
        app_module.main()
    else:
        import streamlit as st
        st.error("`app.py` does not expose a main() function.")


_import_app_and_run()


