from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import zipfile
import subprocess

import pandas as pd
import requests
import streamlit as st
import yaml
import numpy as np


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    # Walk up until we find the workspace root that contains the 'scripts' directory
    for ancestor in here.parents:
        if (ancestor / "scripts").is_dir():
            return ancestor
    # Fallback to three levels up from /pages/ which is typically the project root
    return here.parents[3]


def import_crav4() -> Optional[object]:
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


def validate_owner_repo(owner_repo: str) -> Optional[Tuple[str, str]]:
    owner_repo = owner_repo.strip().strip('/')
    if not owner_repo or '/' not in owner_repo:
        return None
    owner, repo = owner_repo.split('/', 1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo or any(c in repo for c in (' ', '\\')):
        return None
    return owner, repo


def download_github_repo_zip(owner: str, repo: str, ref: Optional[str] = None) -> Path:
    """Robustly download a GitHub repo zip using codeload; fall back to API zipball.

    - If ref is not provided, resolve default branch via GitHub API, then use codeload
    - Supports optional GitHub token via GITHUB_TOKEN env var (for higher rate limits)
    """
    token = get_github_token()
    headers_api = {"Accept": "application/vnd.github+json", "User-Agent": "crav4-demo"}
    if token:
        headers_api["Authorization"] = f"Bearer {token}"

    # Resolve ref (default branch) if not provided
    resolved_ref = (ref or "").strip() or None
    if not resolved_ref:
        with st.spinner("Resolving default branch..."):
            info_resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}", headers=headers_api, timeout=30
            )
        if info_resp.status_code == 200:
            try:
                resolved_ref = info_resp.json().get("default_branch") or None
            except Exception:
                resolved_ref = None
        elif info_resp.status_code == 404:
            raise RuntimeError(f"Repository not found: {owner}/{repo}")
        if not resolved_ref:
            # Fallback guesses
            for guess in ("main", "master"):
                test = requests.get(
                    f"https://codeload.github.com/{owner}/{repo}/zip/{guess}", timeout=60
                )
                if test.status_code == 200:
                    resolved_ref = guess
                    break
            if not resolved_ref:
                # As a last resort, try API zipball without ref (default branch)
                api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
                with st.spinner("Downloading repository zip (API)..."):
                    resp = requests.get(api_url, headers=headers_api, timeout=60)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"GitHub API error {resp.status_code}: {resp.text[:200]}"
                    )
                tmp_dir = Path(tempfile.mkdtemp(prefix="crav4_repo_"))
                with zipfile.ZipFile(BytesIO(resp.content)) as zf:
                    zf.extractall(tmp_dir)
                subdirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
                return subdirs[0] if len(subdirs) == 1 else tmp_dir

    # Prefer codeload for direct zip streaming
    for candidate in (
        f"https://codeload.github.com/{owner}/{repo}/zip/{resolved_ref}",
        f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{resolved_ref}",
        f"https://codeload.github.com/{owner}/{repo}/zip/refs/tags/{resolved_ref}",
    ):
        with st.spinner("Downloading repository zip..."):
            resp = requests.get(candidate, timeout=60)
        if resp.status_code == 200:
            tmp_dir = Path(tempfile.mkdtemp(prefix="crav4_repo_"))
            with zipfile.ZipFile(BytesIO(resp.content)) as zf:
                zf.extractall(tmp_dir)
            subdirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
            return subdirs[0] if len(subdirs) == 1 else tmp_dir

    # Fallback to API zipball with ref
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/zipball/{resolved_ref}"
        if resolved_ref
        else f"https://api.github.com/repos/{owner}/{repo}/zipball"
    )
    with st.spinner("Downloading repository zip (API fallback)..."):
        resp = requests.get(api_url, headers=headers_api, timeout=60)
    if resp.status_code != 200:
        hint = (
            "Check owner/repo and ref. If private or rate-limited, set GITHUB_TOKEN env var or add github_tokens in config.yaml."
        )
        raise RuntimeError(
            f"GitHub download failed ({resp.status_code}). {hint} Details: {resp.text[:200]}"
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="crav4_repo_"))
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        zf.extractall(tmp_dir)
    subdirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else tmp_dir


def clone_github_repo(owner: str, repo: str, ref: Optional[str] = None) -> Path:
    """Clone a GitHub repository using a shallow clone similar to collectors.

    - Uses `git clone --depth 1` and optional `--branch <ref>`
    - Clones into a temporary directory and returns the repo path
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="crav4_repo_clone_"))
    repo_dir = tmp_dir / f"{owner}_{repo}"
    args = [
        "git",
        "clone",
        "--depth",
        "1",
    ]
    if ref:
        args += ["--branch", ref]
    token = get_github_token()
    if token:
        clone_url = f"https://oauth2:{token}@github.com/{owner}/{repo}.git"
    else:
        clone_url = f"https://github.com/{owner}/{repo}.git"
    args += [clone_url, str(repo_dir)]

    with st.spinner("Cloning repository (shallow)..."):
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Git is not installed or not in PATH.") from exc

    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()[:300]}")

    if not repo_dir.exists() or not any(repo_dir.iterdir()):
        raise RuntimeError("Clone completed but repository directory is empty.")

    return repo_dir


def get_github_token() -> Optional[str]:
    """Return a GitHub token from env or config.yaml (github_tokens[0]) if available."""
    env_token = os.getenv("GITHUB_TOKEN")
    if env_token:
        return env_token.strip()
    try:
        root = get_project_root()
        cfg_path = root / "config.yaml"
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            tokens = data.get("github_tokens") or []
            if isinstance(tokens, list) and tokens:
                return str(tokens[0]).strip()
    except Exception:
        pass
    return None


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


def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    try:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_results(results: Dict, owner: str, repo: str) -> Path:
    root = get_project_root()
    out_dir = root / "enhanced_analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{owner}_{repo}".replace(' ', '_')
    out_file = out_dir / f"crav4_demo_{safe_name}.json"
    out_file.write_text(json.dumps(_make_json_safe(results), indent=2), encoding="utf-8")
    return out_file


def main() -> None:
    st.title("CRAv4 Demo")
    st.write("Analyze a public GitHub repository using Comprehensive Repository Analyzer v4.")

    AnalyzerClass = import_crav4()
    if AnalyzerClass is None:
        st.stop()

    with st.sidebar:
        st.header("GitHub Repository")
        owner_repo = st.text_input("owner/repo", placeholder="octocat/Hello-World")
        ref = st.text_input("Optional ref (branch, tag, or commit)", value="")
        run_button = st.button("Fetch and Analyze", type="primary")

    if run_button:
        pair = validate_owner_repo(owner_repo)
        if pair is None:
            st.error("Enter a valid repository as owner/repo (e.g., microsoft/vscode)")
            st.stop()

        owner, repo = pair

        try:
            # Prefer git clone to mirror collector behavior; fall back to zip
            repo_path = clone_github_repo(owner, repo, ref.strip() or None)
        except Exception:
            try:
                repo_path = download_github_repo_zip(owner, repo, ref.strip() or None)
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                st.stop()

        with st.status("Running analysis...", expanded=True) as status:
            st.write(f"Analyzing: {owner}/{repo}")
            try:
                analyzer = AnalyzerClass()
                results = analyzer.analyze_repository(str(repo_path))
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
            out_file = save_results(results, owner, repo)
            st.success(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
