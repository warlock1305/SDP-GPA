from __future__ import annotations

from pathlib import Path
import streamlit as st


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "scripts").is_dir():
            return ancestor
    return here.parents[2]


def main() -> None:
    st.set_page_config(page_title="SDP GPA - Analysis Web UI", page_icon="🔎", layout="wide")
    st.title("SDP GPA - Analysis Web UI")
    st.write("Use the pages in the left sidebar to navigate.")

    st.subheader("Quick Links")
    st.page_link("pages/1_Results_Viewer.py", label="Results Viewer", icon="📁")
    st.page_link("pages/2_CRAv4_Demo.py", label="CRAv4 Demo", icon="🧠")

    st.divider()
    st.caption(f"Project root: {get_project_root()}")


if __name__ == "__main__":
    main()


