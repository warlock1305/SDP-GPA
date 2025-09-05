# Smart Repository Classifier (SRC)
## Technical Documentation & Analysis Report

---

## Executive Summary

The Smart Repository Classifier (SRC) is a hybrid, production-oriented system that classifies repositories into high-level categories by combining deterministic rules with machine-learning predictions. It builds on the feature-extraction stack established in CRAv3/CRAv4 (AST, CodeBERT semantics, keywords, structure) and introduces a robust orchestration layer that:

- Prioritizes highly precise rule-based decisions when clear signals exist
- Falls back to ML (RandomForest + StandardScaler, with optional SelectKBest) when rules are inconclusive and models are ready
- Provides interpretable confidence by explicitly blending semantic, framework, keyword, and structural evidence

---

## 1. System Architecture

### 1.1 Core Components

- RuleBasedClassifier
  - Encodes deterministic rules for categories such as `cli_tool`, `data_science`, `web_application`, `mobile_app`, `game_development`, `library`
  - Uses file/content fingerprints, package metadata cues (e.g., `console_scripts`), and domain keywords (e.g., web/frontend assets)

- FeatureSelector (optional)
  - SelectKBest-based dimensionality reduction to stabilize downstream ML and reduce noise

- SmartRepositoryClassifier (orchestrator)
  - Extracts and consolidates features (via CRAv3/CRAv4 stack)
  - Runs rules first; if inconclusive, checks `_is_ml_ready()` and invokes ML
  - Blends rule/semantic/ML signals into a final classification + confidence

### 1.2 Data Flow

```
Repository → Feature Extraction (AST + CodeBERT + Keywords + Structure)
           → RuleBased Classification (decisive? return) 
           → ML Gate (_is_ml_ready?) → RandomForest prediction
           → Confidence Synthesis (semantic + framework + keywords + structure + ML)
           → Result (category, confidence, indicators)
```

---

## 2. Feature Foundation

SRC reuses the CRAv3/CRAv4 feature stack:
- AST metrics (AstMiner with regex fallbacks): path complexity, node-type diversity, function/class counts
- CodeBERT semantics: repository embedding, moments (mean/std/kurtosis), semantic diversity/coherence, significant dimensions
- Keyword analysis: enhanced analyzer with normalized, multi-bucket lexicons and structural cues
- Structure features: file and directory counts, language distribution, CI/CD, Docker, docs, dependencies

---

## 3. Rule-Based Classification

### 3.1 Rule Design Principles
- Precision over recall: trigger only on strong, specific indicators (e.g., `console_scripts` in Python packaging for CLI)
- Multi-signal validation: require a combination of indicators (keywords + files + metadata) to avoid false positives
- Domain fingerprints:
  - CLI: `console_scripts`, `bin/`, CLI libraries (`click`, `argparse`, `typer`) 
  - Web: frontend assets (React/Angular/Vue), `package.json`, server frameworks (Django/Flask/Express)
  - Game: engines/assets (Phaser/melonJS), game-related keywords, graphics/media presence
  - Mobile: RN/Expo templates, Android/iOS project scaffolding
  - Library: packaging metadata, tests, docs, modular structure
  - Data Science: notebooks, ML/DS libraries, datasets, pipelines

### 3.2 Rule Execution
- `_extract_rule_features(repo_path)`: scans the repository for rule inputs
- `_evaluate_rule(features, rule)`: computes a score per rule; top-scoring category wins if above threshold
- Fast path: If a rule is decisive, return classification without invoking ML

---

## 4. Machine Learning Layer

### 4.1 Readiness Gate
- `_is_ml_ready()` ensures `StandardScaler` and RandomForest models are fitted before inference
- If not ready, fallback to rules to maintain availability

### 4.2 Models
- RandomForest classifiers trained on consolidated features (AST + semantics + keywords + structure + significant dimensions)
- StandardScaler for numerical stability
- Optional: SelectKBest to reduce dimensionality and highlight discriminative features

### 4.3 Inference Path
- `_ml_classify(repo_path, arch_type)`
  - Builds feature vector from extracted dictionaries
  - Applies scaler → RF prediction → probability distribution
  - Produces ML category and confidence, later blended with rule/semantic evidence

---

## 5. Confidence Synthesis

Confidence is calculated with explicit, interpretable weights (category-specific):
- Semantic indicators (from significant dimensions and semantic analysis)
- Framework detection (React/Angular/Vue, Django/Flask/Express)
- Keyword strength (domain-specific lexicons)
- Structural indicators (e.g., console_scripts, frontend files, packaging metadata)
- ML probability (when used) and category score (from significant dims)

This produces category-specific `_calculate_*_confidence` functions that yield calibrated, bounded scores.

---

## 6. Usage Examples

### 6.1 Basic Classification
```python
from scripts.analysis.smart_repository_classifier import SmartRepositoryClassifier

cls = SmartRepositoryClassifier(models_dir="ml_models")
result = cls.classify("/path/to/repository")
print(result["category"], result["confidence"])  # e.g., web_application, 0.83
```

### 6.2 Training (optional)
- If training utilities are exposed, you can prepare datasets via the `dataset/` layout and fit models for your environment.
- Persist fitted models and scaler to `ml_models/` so `_is_ml_ready()` passes in production.

---

## 7. Design Decisions & Rationale

- Rules first, ML second: reduces latency and misclassifications on strongly-typed repos
- Dataset-driven semantics: significant CodeBERT dimensions stabilize predictions and category scores
- Interpretable confidence: explicit weights reflect empirical separability seen in PCA/t-SNE and pairwise tests
- Graceful degradation: when ML is not ready or embeddings fail, rules and structure still yield usable results

---

## 8. Performance & Reliability

- Lazy model loading (inherited from CRAv3/CRAv4 for heavy deps)
- Batching for embedding, with CUDA support when available
- Timeouts and fallbacks for AST extraction
- Safe empty outputs ensure pipeline continuity

---

## 9. Future Enhancements

- Learn weights for confidence blending (e.g., isotonic regression) while preserving interpretability
- Expand rule library with provenance (which evidence triggered classification)
- Domain adapters or light fine-tuning on top of CodeBERT for challenging category boundaries
- Language-specific AST enrichments to reduce regex reliance

---

## 10. Summary

The Smart Repository Classifier provides a pragmatic, hybrid strategy for repository classification, tuned by empirical dataset analyses. It leverages the robustness of deterministic rules and the adaptability of ML, backed by a rich feature stack. The outcome is an interpretable, resilient, and accurate classifier well-suited for heterogeneous, real-world codebases.

