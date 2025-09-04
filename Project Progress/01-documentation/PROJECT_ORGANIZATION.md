# Project Organization & Script Documentation

## 📁 Directory Structure

```
SDP GPA/
├── scripts/
│   ├── collectors/           # Repository collection scripts
│   ├── extraction/          # Feature extraction scripts
│   ├── analysis/            # Analysis and processing scripts
│   └── models/              # Machine learning models
├── data/                    # Data files (progress, metadata)
├── dataset/                 # Collected repositories
├── models/                  # Trained models
├── config.yaml             # Configuration file
└── PROJECT_SUMMARY.md      # Project overview
```

## 🔧 Scripts Overview

### **📥 Collectors (`scripts/collectors/`)**

#### `corrected_single_contributor_collector.py` ⭐ **CURRENT WORKING VERSION**
- **Purpose**: Collects properly categorized single-contributor repositories
- **Features**: 
  - Skips already cloned repositories
  - Proper categorization (web_app, data_science, library, cli_tool, educational)
  - Debug logging
  - 100% success rate for verified repositories
- **Status**: ✅ **ACTIVE** - Used for current dataset collection
- **Output**: 29 repositories across 5 categories

#### `collector.py` 📚 **ORIGINAL BASE**
- **Purpose**: Original repository collector with API integration
- **Features**: 
  - GitHub API integration
  - Token rotation
  - Rate limiting
  - File content extraction
- **Status**: 📖 **REFERENCE** - Base implementation for other collectors

### **🔍 Extraction (`scripts/extraction/`)**

#### `extract_ast_features_for_analysis.py`
- **Purpose**: Extract AST features from collected repositories
- **Features**: 
  - Multi-language AST extraction
  - Feature vector generation
  - Path-based representations
- **Status**: 🔄 **READY** - Next step in pipeline

#### `extract_codebert_embeddings.py`
- **Purpose**: Generate CodeBERT embeddings for code analysis
- **Features**: 
  - Multi-language code embeddings
  - Semantic representation
  - Batch processing
- **Status**: 🔄 **READY** - Next step in pipeline

### **📊 Analysis (`scripts/analysis/`)**

#### `enhanced_keyword_analyzer.py`
- **Purpose**: Keyword-based analysis of repositories
- **Features**: 
  - Pattern detection
  - Educational content identification
  - Quality metrics
- **Status**: 🔄 **READY** - For comprehensive analysis

### **🤖 Models (`scripts/models/`)**

#### `comprehensive_random_forest_classifier.py`
- **Purpose**: Multi-modal Random Forest classifier
- **Features**: 
  - AST + CodeBERT + Keyword features
  - Architectural pattern classification
  - Multi-output prediction
- **Status**: 🔄 **READY** - For training on collected dataset

## 🗑️ Removed Redundant Scripts

The following scripts were **removed** as they were redundant or replaced by better versions:

### **❌ Deleted Collectors:**
1. `single_contributor_collector.py` - Early attempt, replaced by `corrected_single_contributor_collector.py`
2. `public_repository_collector.py` - Alternative approach, not used
3. `enhanced_collector.py` - Enhancement of original, not used
4. `real_single_contributor_collector.py` - Replaced by corrected version
5. `small_single_contributor_collector.py` - Replaced by corrected version
6. `repository_collector.py` - Different approach, not used

### **📊 Why They Were Removed:**
- **Duplication**: Multiple scripts doing the same thing
- **Better Alternatives**: `corrected_single_contributor_collector.py` is superior
- **Maintenance**: Fewer files to maintain
- **Confusion**: Clear which script to use

## 📈 Current Status

### **✅ Completed:**
- Repository collection (29 repos across 5 categories)
- Dataset organization
- Script cleanup and organization
- CLI tools population

### **🔄 Next Steps:**
1. **AST Feature Extraction**: Run `extract_ast_features_for_analysis.py`
2. **CodeBERT Embeddings**: Run `extract_codebert_embeddings.py`
3. **Keyword Analysis**: Run `enhanced_keyword_analyzer.py`
4. **Model Training**: Run `comprehensive_random_forest_classifier.py`

## 🎯 Usage Instructions

### **To Collect More Repositories:**
```bash
python scripts/collectors/corrected_single_contributor_collector.py
```

### **To Extract AST Features:**
```bash
python scripts/extraction/extract_ast_features_for_analysis.py
```

### **To Generate CodeBERT Embeddings:**
```bash
python scripts/extraction/extract_codebert_embeddings.py
```

### **To Train the Model:**
```bash
python scripts/models/comprehensive_random_forest_classifier.py
```

## 📋 Data Files

### **Progress Files (`data/`):**
- `corrected_single_contributor_progress.json` - Current collection progress
- `corrected_single_contributor_metadata.csv` - Repository metadata

### **Dataset (`dataset/`):**
- `web_application/` - 10 repositories
- `data_science/` - 3 repositories  
- `library/` - 7 repositories
- `cli_tool/` - 7 repositories
- `educational/` - 13 repositories

## 🔧 Configuration

### **`config.yaml`:**
- Supported languages
- GitHub API configuration
- File extensions
- Collection limits

## 📚 Documentation

### **`PROJECT_SUMMARY.md`:**
- Complete project overview
- Technical details
- Implementation status
- Next steps

---

**🎉 Result**: Clean, organized project structure with clear separation of concerns and no redundant scripts!
