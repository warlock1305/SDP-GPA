import os
import subprocess
import tempfile
import yaml

# === CONFIGURATION ===
ASTMINER_JAR = os.path.abspath("astminer-0.9.0/build/libs/astminer.jar")
DATASET_ROOT = os.path.abspath("dataset")
OUTPUT_ROOT = os.path.abspath("ExtractedPaths")

# Supported file extensions → parser names in AstMiner
EXT_TO_LANG = {
    ".java": "java",
    ".py": "py",
    ".js": "js",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".kt": "kotlin",
    ".kts": "kotlin"
}


def detect_project_languages(project_path):
    """Detect all supported languages present in a project."""
    langs = set()
    for root, _, files in os.walk(project_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXT_TO_LANG:
                langs.add(EXT_TO_LANG[ext])
    return langs


def run_astminer(lang, project_path, output_path):
    """Run AstMiner for a given language/project using a temporary config file."""
    os.makedirs(output_path, exist_ok=True)

    # Build config dictionary
    config = {
        "inputDir": project_path,
        "outputDir": output_path,
        "parser": {
            "name": "antlr",
            "languages": [lang]
        },
        "filters": [
            {
                "name": "by tree size",
                "maxTreeSize": 1000
            }
        ],
        "label": {
            "name": "file name"
        },
        "storage": {
            "name": "code2seq",
            "length": 9,
            "width": 2
        },
        "numOfThreads": 1
    }

    # Save config to a temporary YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    try:
        cmd = ["java", "-jar", ASTMINER_JAR, tmp_path]
        print(f"[RUN] {project_path} [{lang}] → {output_path}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] {project_path} [{lang}] → {result.stderr.strip()}")
            return False
        else:
            print(f"[SUCCESS] {project_path} [{lang}] → {output_path}")
            return True
    except Exception as e:
        print(f"[ERROR] {project_path} [{lang}] → {str(e)}")
        return False
    finally:
        # Remove the temporary config file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_all_repos():
    """Walk through dataset/user/repo and process each repo for all detected languages."""
    print("=== Multi-language AST extraction started ===")
    
    # Check if astminer.jar exists
    if not os.path.exists(ASTMINER_JAR):
        print(f"[ERROR] astminer.jar not found at {ASTMINER_JAR}")
        print("Please ensure astminer is properly installed")
        return

    processed_count = 0
    error_count = 0

    for user in os.listdir(DATASET_ROOT):
        user_path = os.path.join(DATASET_ROOT, user)
        if not os.path.isdir(user_path):
            continue

        for repo in os.listdir(user_path):
            repo_path = os.path.join(user_path, repo)
            if not os.path.isdir(repo_path):
                continue

            print(f"\n[PROCESSING] {user}/{repo}")
            langs = detect_project_languages(repo_path)
            if not langs:
                print(f"[SKIP] {user}/{repo} → No supported languages found")
                continue

            for lang in langs:
                output_path = os.path.join(OUTPUT_ROOT, user, repo, lang)
                success = run_astminer(lang, repo_path, output_path)
                if success:
                    processed_count += 1
                else:
                    error_count += 1

    print(f"\n=== Summary ===")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {processed_count + error_count}")
    print("=== Done ===")


if __name__ == "__main__":
    process_all_repos()
