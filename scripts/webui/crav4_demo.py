from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


def get_project_root() -> Path:
    # This file lives at scripts/webui/crav4_demo.py → root is two levels up
    return Path(__file__).resolve().parents[2]


def import_crav4() -> Optional[object]:
    """Dynamically import CRAv4 analyzer class.

    Returns None if import fails (e.g., missing dependencies), and shows an error in UI.
    """
    root = get_project_root()
    analysis_dir = root / "scripts" / "analysis"
    sys.path.append(str(analysis_dir))
    try:
        from comprehensive_repository_analyzer_v4 import (  # type: ignore
            ComprehensiveRepositoryAnalyzerV4,
        )
        return ComprehensiveRepositoryAnalyzerV4
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Failed to import CRAv4 analyzer. Ensure required packages are installed "
            "and the file exists at `scripts/analysis/comprehensive_repository_analyzer_v4.py`.\n\n"
            f"Error: {exc}"
        )
        return None


def list_candidate_repo_roots(base_dir: Path) -> List[Path]:
    """Suggest common directories that may contain repositories to analyze."""
    candidates = [
        base_dir,
        base_dir / "design_pattern_dataset" / "real_repositories",
        base_dir / "design_pattern_dataset" / "small_repositories",
        base_dir / "dataset",
        base_dir / "AI-Github-Profile-Analyser",
    ]
    return [p for p in candidates if p.exists() and p.is_dir()]


def list_immediate_subdirs(parent: Path) -> List[Path]:
    try:
        return sorted([p for p in parent.iterdir() if p.is_dir()])
    except Exception:
        return []


def render_summary_metrics(results: Dict) -> None:
    quality = results.get("quality_assessment", {}).get("overall_score", 0.0)
    files = results.get("features", {}).get("structure_features", {}).get("total_files", 0)
    functions = results.get("features", {}).get("ast_features", {}).get("function_count", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Quality", f"{quality:.3f}")
    with col2:
        st.metric("Files", f"{files:,}")
    with col3:
        st.metric("Functions", f"{int(functions):,}")


def render_detected_patterns(results: Dict) -> None:
    arch = results.get("architecture_analysis", {})
    patterns = arch.get("detected_patterns", [])
    conf = arch.get("pattern_confidence", {})
    if not patterns:
        st.info("No patterns detected.")
        return
    df = pd.DataFrame({
        "pattern": patterns,
        "confidence": [conf.get(p, 0.0) for p in patterns],
    })
    st.dataframe(df, width="stretch")


def render_quality_tables(results: Dict) -> None:
    qa = results.get("quality_assessment", {})
    with st.expander("Quality Assessment Details", expanded=False):
        for section_name in [
            "code_quality",
            "architecture_quality",
            "documentation_quality",
            "maintainability",
        ]:
            section = qa.get(section_name, {})
            if section:
                st.markdown(f"**{section_name.replace('_', ' ').title()}**")
                st.dataframe(pd.DataFrame([section]).T.rename(columns={0: "score"}), width="stretch")


def render_language_breakdown(results: Dict) -> None:
    structure = results.get("features", {}).get("structure_features", {})
    lang_counts = structure.get("language_counts", {})
    if lang_counts:
        df = pd.DataFrame(
            sorted(((k, v) for k, v in lang_counts.items()), key=lambda x: -x[1]),
            columns=["language", "files"],
        )
        st.bar_chart(df, x="language", y="files")


def save_results(results: Dict, repo_path: Path) -> Path:
    root = get_project_root()
    out_dir = root / "enhanced_analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = repo_path.name.replace(' ', '_')
    out_file = out_dir / f"crav4_demo_{safe_name}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out_file


def main() -> None:
    st.title("CRAv4 Demo")
    st.write("Run the Comprehensive Repository Analyzer v4 on a local folder and preview results.")

    root = get_project_root()
    AnalyzerClass = import_crav4()
    if AnalyzerClass is None:
        st.stop()

    with st.sidebar:
        st.header("Target Repository")
        repo_roots = list_candidate_repo_roots(root)
        root_labels = [str(p.relative_to(root)) if p != root else "." for p in repo_roots]
        selected_root_label = st.selectbox("Select a base folder", options=root_labels, index=0)
        selected_root = repo_roots[root_labels.index(selected_root_label)]

        subdirs = list_immediate_subdirs(selected_root)
        sub_labels = ["(use base folder)"] + [d.name for d in subdirs]
        selected_sub = st.selectbox("Select a subfolder (optional)", options=sub_labels, index=0)

        if selected_sub != "(use base folder)":
            target_path = selected_root / selected_sub
        else:
            target_path = selected_root

        manual_path = st.text_input(
            "Or enter a custom absolute path",
            value=str(target_path),
        )
        target_repo = Path(manual_path)

        run_button = st.button("Analyze with CRAv4", type="primary")

    if run_button:
        if not target_repo.exists() or not target_repo.is_dir():
            st.error(f"Path does not exist or is not a directory: {target_repo}")
            st.stop()

        with st.status("Running analysis...", expanded=True) as status:
            st.write(f"Analyzing: {target_repo}")
            try:
                analyzer = AnalyzerClass()
                results = analyzer.analyze_repository(str(target_repo))
                status.update(label="Analysis complete", state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label="Analysis failed", state="error")
                st.exception(exc)
                st.stop()

        render_summary_metrics(results)
        st.subheader("Detected Patterns")
        render_detected_patterns(results)
        render_quality_tables(results)
        st.subheader("Language Breakdown")
        render_language_breakdown(results)

        with st.expander("Raw Summary", expanded=False):
            st.json(results.get("summary", {}), expanded=False)

        if st.toggle("Save results JSON", value=True):
            out_file = save_results(results, target_repo)
            st.success(f"Saved to {out_file}")


if __name__ == "__main__":
    main()


