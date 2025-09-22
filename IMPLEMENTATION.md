## IMPLEMENTATION

### Overview

This section describes the implementation of the analysis suite, including CRAv4, the new design-pattern detector, and the Streamlit-based web UI. It covers the programming languages, libraries, tooling, runtime environment, external services/APIs, decision logic, features, and how to run the system locally. Screenshot placeholders are provided for documentation.


### Technology Stack

- Languages
  - Python 3.11/3.12 (primary application and analyzers)
  - Java 11+ (AstMiner runner for AST extraction)

- Core Python libraries
  - Analysis and data: `numpy`, `pandas`, `scikit-learn`, `pyyaml`
  - NLP/embeddings: `transformers` (CodeBERT), `torch` (backend for CodeBERT)
  - UI: `streamlit`, `pandas`, `pyarrow`
  - Utilities: `requests` (GitHub fetch), `zipfile`, `subprocess`

- External tools
  - Git (for shallow clones `--depth 1`) in CRAv4 demo page
  - Java (JDK 11+) and AstMiner jar: `astminer-0.9.0/build/libs/astminer.jar`

- Services/APIs
  - GitHub codeload + REST API (zip snapshots, repo metadata, default branch resolution)
  - Hugging Face model hub (automatic download of `microsoft/codebert-base` weights and tokenizer)

- OS/Hardware
  - Targeted and tested on Windows 10/11; works on Linux/macOS with Java+Git available
  - CPU-only is fully supported; GPU (CUDA) optional for faster CodeBERT inference


### Component Architecture

#### CRAv4 (Hybrid Orchestrated Analyzer)

- Entry point: `scripts/analysis/comprehensive_repository_analyzer_v4.py`
- Pipeline stages (per repository path):
  1) AST features (structure)
     - If `astminer.jar` is present and Java is available, runs AstMiner with a guarded timeout and extracts path-context statistics; otherwise falls back to language-aware regex or generic parsing.
  2) Code semantics (CodeBERT)
     - Loads `microsoft/codebert-base` via `transformers` and `torch`, encodes repository files by language, batch-pools file embeddings, and produces repository-level embedding and semantic statistics.
  3) Keyword/content signals
     - Uses enhanced keyword extraction when available; falls back to a robust, curated keyword counter to populate domain and framework indicators.
  4) File structure
     - Computes counts (files, directories, source files), depth, language diversity, and boolean flags (tests/docs/config/dependencies/CI-Docker).
  5) Fusion + decisions
     - Combines AST + semantic + keyword + structure feature sets; computes category scores from CodeBERT significant dimensions; applies rule-augmented indicators to detect patterns; computes confidences.

- Features used (high-level)
  - AST: average/max path length, path variety, node-type diversity, complexity proxy, nesting depth, function/class/interface counts, total AST nodes, branching factor
  - Semantic (from embeddings): mean/std/max/min/range, skewness, kurtosis, semantic diversity/coherence
  - Significant dimensions: curated indices from dataset analyses for pairs (e.g., CLI vs DS, Web vs Library, DS vs Web, etc.)
  - Keyword buckets: web, cli, data_science, library, mobile, game, testing, database, cloud, totals/diversity
  - Structure: file and directory stats, language distribution, presence of tests/docs/config/dependencies, CI/CD, Docker

- Decision and confidence policy
  - Category scores: computed from CodeBERT significant dimensions per category pair; scores normalized by number of contributing dims
  - Indicators (pattern-specific): OR-combinations of semantic cues, keyword thresholds, structural flags (e.g., frontend files, CI/CD, Docker), and category-score gates
  - Confidence: weighted sum from semantic evidence, keyword strength, structural flags, and category score contribution; clipped to [0,1]
  - Fail-safe operation: when AstMiner or CodeBERT is unavailable, the analyzer falls back to simpler features while preserving output schema; decisions degrade gracefully

- Performance engineering
  - Lazy model load for CodeBERT
  - Batch inference with deterministic max length (512) and pooling
  - Java subprocess timeouts for AST to avoid hangs on very large trees
  - GPU memory cleanup when CUDA is available


#### New Design-Pattern Detector (Structural/Relational)

- Location: `d_p_det/` (data, models, results, src, tests)
- Design principle: prioritize structural and relational signatures over raw numeric ratios, in line with project requirements
  - Focus on graph/structural cues (e.g., inheritance relationships, call chains, collaborator roles) and path/relationship schematics
  - Encodes patterns as structural predicates and relational templates that must be matched in code topology
  - Produces per-pattern evidence with interpretable provenance (which relationships matched)

- Typical workflow
  1) Code parsing to graph/CPG/AST-like structures (tooling may use ASTMiner outputs and additional parsers)
  2) Rule/template application over graph relations (e.g., class participates in delegation; observer registration; adapter roles)
  3) Candidate scoring and heuristic disambiguation to reduce false positives (e.g., verify role multiplicity, directionality)
  4) Pattern report with structural evidence and confidence

- Notes
  - Emphasis on structural schematics aligns with the project’s preference to avoid over-reliance on numeric counts
  - Detector can be used standalone or as an auxiliary module to CRAv4 analysis artifacts


#### Streamlit Web UI

- Location
  - Home: `scripts/webui/Home.py`
  - Pages: `scripts/webui/pages/1_Results_Viewer.py` and `scripts/webui/pages/2_CRAv4_Demo.py`
  - Viewer module: `scripts/webui/app.py` (used by 1_Results_Viewer)

- Dependencies
  - `streamlit`, `pandas`, `requests`, `numpy`, `pyyaml`, `pyarrow`
  - Uses system `git` if available (preferred) and falls back to GitHub zip snapshot

- How it works
  - Results Viewer: browses CSV/JSON/PNG artifacts under known result directories; displays dataframes/images inline; paths rendered relative to project root
  - CRAv4 Demo:
    - Input: single `owner/repo` (and optional `ref`)
    - Fetch: attempts a shallow clone `git clone --depth 1` (optionally `--branch ref`), or downloads a codeload/API zip snapshot
    - Analyze: runs CRAv4 end to end on the fetched repository path
    - Save: writes JSON to `enhanced_analysis_results/crav4_demo_<owner>_<repo>.json` with JSON-safe serialization

- Local execution (recommended)
  - Use the same Python interpreter that has CRAv4 dependencies; example:
    - `python -m pip install -r requirements.txt`
    - `python -m streamlit run scripts/webui/Home.py --server.port 8501`


### Implementation Environment

- Required software
  - Python 3.11 or 3.12 (+ pip)
  - Java JDK 11+ (for AstMiner path-extraction mode)
  - Git (for shallow clone path; optional if zip fallback is used)
  - Internet access (first-run model weights; GitHub downloads)

- Optional
  - NVIDIA GPU + CUDA (for faster CodeBERT inference)

- Configuration
  - `config.yaml` can hold `github_tokens` for higher GitHub API limits; the CRAv4 demo reads the first token if present


### APIs and Services

- GitHub
  - Resolve default branch: `GET https://api.github.com/repos/{owner}/{repo}`
  - Zip snapshot download (prefer): `https://codeload.github.com/{owner}/{repo}/zip/{ref}`
  - API zipball fallback: `GET https://api.github.com/repos/{owner}/{repo}/zipball/{ref?}`
  - Auth (optional): `GITHUB_TOKEN` environment variable or first token from `config.yaml`

- Hugging Face
  - Model: `microsoft/codebert-base`
  - Auto-fetched by `transformers` on first use


### How CRAv4 Judges and Predicts

1) Evidence extraction
   - Structural metrics (ASTMiner or fallbacks)
   - Semantic signals (CodeBERT embeddings and statistics)
   - Keyword/domain heuristics (enhanced or curated fallback)
   - Organization and tooling indicators (tests/docs/deps/CI/Docker)

2) Category scores from significant dimensions
   - Aggregates per-pair curated dimensions (e.g., CLI vs DS, Web vs Library) and normalizes their contributions
   - Produces calibrated scores used as a soft prior for pattern decisions

3) Pattern detection
   - For each pattern (web_application, library, cli_tool, data_science, mobile_app, game_development, etc.), evaluate logical indicators combining:
     - Semantic evidence ≥ thresholds
     - Keyword counts ≥ thresholds
     - Structural/organizational flags
     - Category score gates

4) Confidence estimation
   - Adds weighted contributions: semantic > framework/API cues > keywords > structure > category score; clipped to [0,1]
   - Returns interpretable confidences per detected pattern

5) Quality and characteristics
   - Quality aggregates normalized complexity, structural and organization scores, semantic consistency, and maintainability cues
   - Programmer characteristics (experience/style/best practices) estimate based on feature thresholds and composite indices


### Running Locally

1) Ensure Java and (optionally) Git are installed and on PATH; verify with `java -version` and `git --version`
2) Install Python dependencies using the interpreter that already has CRAv4 deps (torch/transformers):
   - `python -m pip install -r requirements.txt`
3) Launch the UI:
   - `python -m streamlit run scripts/webui/Home.py --server.port 8501`
4) CRAv4 demo page: enter `owner/repo` and optional `ref`, then click “Fetch and Analyze”. Results are saved under `enhanced_analysis_results/`.


### Screenshot Placeholders

Insert actual screenshots at the suggested paths and link them here:

1) Web UI Home
   - `docs/screenshots/webui_home.png`
   - Description: Landing page with links to Results Viewer and CRAv4 Demo

2) Results Viewer browsing
   - `docs/screenshots/results_viewer_table.png`
   - Description: CSV rendered in a table; JSON/PNG rendering where applicable

3) CRAv4 Demo input and analysis
   - `docs/screenshots/crav4_demo_input.png`
   - Description: owner/repo input and optional ref; analysis in progress (status panel)

4) CRAv4 Demo results
   - `docs/screenshots/crav4_demo_results.png`
   - Description: summary metrics, detected patterns with confidences, quality tables, language breakdown bar chart


### Notes and Limitations

- ASTMiner mode requires Java and a built jar; otherwise, structural fallbacks are used
- Very large repositories may exceed the AST timeout; increase it in the analyzer or analyze a smaller ref/branch
- First CodeBERT use downloads model weights; requires network and may take a few minutes
- Keep Streamlit running under the same Python interpreter as CRAv4 to ensure `torch`/`transformers` availability


