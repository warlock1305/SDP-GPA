# AST Extraction and Code2Vec Transformation: Complete Guide

## 🎯 **What is an AST (Abstract Syntax Tree)?**

An **Abstract Syntax Tree (AST)** is a tree representation of source code that captures the **structure** and **relationships** between code elements while ignoring formatting, comments, and other details.

### **Simple Example:**

```java
int x = 5 + 3;
```

**AST Representation:**
```
VariableDeclaration
├── Type: int
├── Name: x
└── Initializer: BinaryExpression
    ├── Left: Literal (5)
    ├── Operator: +
    └── Right: Literal (3)
```

---

## 🔍 **Step 1: How AST Values Are Extracted**

### **1.1 Lexical Analysis (Tokenization)**
Your code is broken into tokens:
```
"int x = 5 + 3;" → [int, x, =, 5, +, 3, ;]
```

### **1.2 Syntax Analysis (Parsing)**
Tokens are organized into a parse tree based on grammar rules.

### **1.3 AST Construction**
The parse tree is simplified into an AST by removing unnecessary nodes.

### **1.4 ASTMiner Extraction Process**

Your `extract_multi_lang.py` script uses **ASTMiner** which:

1. **Uses ANTLR parser** to build the AST
2. **Assigns unique IDs** to each node type
3. **Extracts paths** between leaf nodes (tokens)
4. **Stores relationships** in `.c2s` files

**Configuration from your script:**
```yaml
parser:
  name: "antlr"
  languages: ["java"]
storage:
  name: "code2seq"
  length: 9    # Max path length
  width: 2     # Max path width
```

---

## 🛤️ **Step 2: AST Path Extraction**

### **2.1 What is a Path Context?**

A **path context** is a triple: `<start_token, path, end_token>`

- **start_token**: The first token in the path
- **path**: Sequence of AST nodes connecting the tokens
- **end_token**: The last token in the path

### **2.2 Example: Method `add(int a, int b)`**

**Source Code:**
```java
public int add(int a, int b) {
    return a + b;
}
```

**AST Structure:**
```
MethodDeclaration (add)
├── Modifier (public)
├── Type (int)
├── Identifier (add)
├── Parameters
│   ├── Parameter (int a)
│   └── Parameter (int b)
└── Block
    └── ReturnStatement
        └── BinaryExpression (+)
            ├── VariableReference (a)
            └── VariableReference (b)
```

**Extracted Path Contexts:**
```
add,MethodDeclaration→Parameter→VariableDeclaratorId,a
add,MethodDeclaration→Parameter→VariableDeclaratorId,b
a,VariableDeclaratorId→BinaryExpression→VariableReference,b
a,VariableDeclaratorId→BinaryExpression→VariableReference,+
```

### **2.3 Real Example from Your Data**

From your extracted files, here's what a real path context looks like:

```
[],(ArrayBracketPair2)^(Parameter)^(MethodDeclaration)_(BlockStmt)_(ExpressionStmt)_(MethodCallExpr0)_(NameExpr1),comparenumbers
```

**Breaking it down:**
- **Start token**: `[]` (empty array)
- **Path**: `ArrayBracketPair2 → Parameter → MethodDeclaration → BlockStmt → ExpressionStmt → MethodCallExpr0 → NameExpr1`
- **End token**: `comparenumbers`

---

## 🧠 **Step 3: Code2Vec Transformation**

### **3.1 Input Processing**

Code2Vec takes your AST path contexts and transforms them:

**Input:** `add,MethodDeclaration→Parameter→VariableDeclaratorId,a`

**Processing:**
1. **Tokenization**: Convert tokens to IDs
   - `add` → ID: 123
   - `MethodDeclaration→Parameter→VariableDeclaratorId` → ID: 456
   - `a` → ID: 789

2. **Embedding Lookup**: Get vector representations
   - Token `add` → Vector [0.1, -0.3, 0.8, ...] (128-512 dimensions)
   - Path → Vector [0.2, 0.5, -0.1, ...]
   - Token `a` → Vector [-0.4, 0.7, 0.2, ...]

### **3.2 Attention Mechanism**

Code2Vec uses **attention** to weight different paths:

**Example Attention Scores:**
```
Path: method→parameter→variable → Score: 0.85 (high importance)
Path: method→block→statement → Score: 0.45 (medium importance)  
Path: method→modifier→public → Score: 0.12 (low importance)
```

### **3.3 Aggregation**

All weighted path vectors are combined:
```
Final Method Vector = Σ(attention_score × path_vector)
```

---

## 📊 **Step 4: Real Examples from Your Results**

### **4.1 Your Actual Predictions**

From your `Code2VecPredictions/` files:

**Method: `main`**
- **Original name**: `main`
- **Prediction**: `main` (90.85% confidence) ✅
- **Attention path**: `Score: 0.1337 | Path: ArrayBracketPair2→Parameter→MethodDeclaration→BlockStmt→ExpressionStmt→MethodCallExpr0→NameExpr1`

**Method: `print|greeting`**
- **Original name**: `print|greeting`
- **Prediction**: `test|static|method` (34.89% confidence)
- **Attention path**: `Score: 0.1810 | Path: VoidType0→MethodDeclaration→NameExpr1`

### **4.2 What This Tells Us**

1. **High Confidence Predictions**: The model is very confident about common patterns like `main` methods
2. **Attention Focus**: The model pays attention to method structure and parameter relationships
3. **Pattern Recognition**: It learns that certain AST paths are predictive of method names

---

## 🔧 **Step 5: Technical Details**

### **5.1 AST Node Types from Your Data**

From your `node_types.csv`:
```
id,node_type
37,memberDeclaration
124,defaultValue
112,THROW
128,annotationTypeElementDeclaration
...
```

These are the building blocks of your AST paths.

### **5.2 Path Context Format**

Your `.c2s` files contain:
```
method_name path_context1 path_context2 path_context3 ...
```

Each path context: `<start_token,path,end_token>`

### **5.3 Code2Vec Model Architecture**

```
Input: AST Path Contexts
    ↓
Token & Path Embeddings
    ↓
Attention Mechanism
    ↓
Weighted Aggregation
    ↓
Method Vector
    ↓
Output: Method Name Prediction
```

---

## 🎯 **Summary: The Complete Pipeline**

1. **Source Code** → **ASTMiner** → **AST Path Contexts**
2. **AST Path Contexts** → **Code2Vec** → **Token/Path Embeddings**
3. **Embeddings** → **Attention** → **Weighted Vectors**
4. **Weighted Vectors** → **Aggregation** → **Method Vector**
5. **Method Vector** → **Prediction** → **Method Name**

### **Key Insights:**

- **AST extraction** captures the **structural relationships** in your code
- **Path contexts** represent **how different code elements relate** to each other
- **Code2Vec** learns **semantic patterns** from these structural relationships
- **Attention mechanism** identifies **which patterns are most important** for prediction
- **Your results show 46% exact matches**, indicating the model successfully learns meaningful code patterns

This pipeline transforms **structural code information** into **semantic understanding**, allowing the model to predict method names based on their implementation patterns!

