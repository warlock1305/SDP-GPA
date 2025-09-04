import os
import subprocess
import shutil
import tempfile

# === CONFIGURATION ===
JAVA_EXTRACTOR_JAR = os.path.abspath("code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar")
DATASET_ROOT = os.path.abspath("dataset")
OUTPUT_ROOT = os.path.abspath("ExtractedPaths")

def extract_paths(java_file_path, output_file_path):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_path = os.path.join(tmpdir, os.path.basename(java_file_path))
            shutil.copyfile(java_file_path, tmp_file_path)

            result = subprocess.run(
                [
                    "java", "-cp", JAVA_EXTRACTOR_JAR, "JavaExtractor.App",
                    "--dir", tmpdir,
                    "--max_path_length", "8",
                    "--max_path_width", "2",
                    "--max_contexts", "200"
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"[ERROR] {java_file_path} → {result.stderr.strip()}")
                return

            # Write stdout content to output file
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            print(f"[OK] {java_file_path} → {output_file_path}")

    except Exception as e:
        print(f"[EXCEPTION] {java_file_path} → {str(e)}")


def walk_and_extract():
    for root, _, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.endswith(".java"):
                abs_path = os.path.join(root, file)
                # Relative path from DATASET_ROOT, keep subfolder structure
                rel_path = os.path.relpath(abs_path, DATASET_ROOT)
                # Output path mirrors the input path, just under OUTPUT_ROOT with .paths.txt suffix
                output_path = os.path.join(OUTPUT_ROOT, rel_path + ".paths.txt")
                extract_paths(abs_path, output_path)

if __name__ == "__main__":
    print("Extracting AST paths from Java files...")
    walk_and_extract()
    print("Done.")
