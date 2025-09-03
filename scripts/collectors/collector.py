import os
import time
import json
import base64
import requests
import yaml
import logging
import csv

# Load environment variables (optional, not used here)
from dotenv import load_dotenv
load_dotenv()

# Load GitHub tokens and config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Constants
API_URL = "https://api.github.com"
TIMEOUT = 60
MAX_RETRIES = 2
PROGRESS_FILE = "IAMpRoGrEsS.json"
METADATA_FILE = "metadata.csv"

# Loaded from config
GITHUB_TOKENS = config.get("github_tokens", [])
ALLOWED_EXTENSIONS = config.get("allowed_extensions", [])
EXCLUDED_FILENAMES = config.get("excluded_filenames", [])
PARALLEL_LIMIT = config.get("parallel_limit", 10)
MAX_REPOS = config.get("max_repos", 2000)
MAX_TOTAL_SIZE_MB = config.get("max_total_size_mb", 70000)

# Ignored folders (hardcoded for better focus)
IGNORED_FOLDERS = ["node_modules/", ".github/"]

# Token management
current_token_index = 0
headers = {"Authorization": f"token {GITHUB_TOKENS[current_token_index]}"}

def rotate_token():
    global current_token_index
    current_token_index = (current_token_index + 1) % len(GITHUB_TOKENS)
    headers["Authorization"] = f"token {GITHUB_TOKENS[current_token_index]}"
    logging.info(f"Rotated to token {current_token_index + 1}")

def check_rate_limit(token_index):
    url = f"{API_URL}/rate_limit"
    headers_local = {"Authorization": f"token {GITHUB_TOKENS[token_index]}"}
    response = requests.get(url, headers=headers_local)
    data = response.json()
    return data["resources"]["core"]["remaining"], data["resources"]["core"]["reset"]

def handle_rate_limit(token_index):
    remaining, reset_time = check_rate_limit(token_index)
    if remaining == 0:
        wait_time = reset_time - int(time.time()) + 5
        logging.info(f"Token {token_index + 1} exhausted, sleeping for {wait_time} seconds...")
        time.sleep(wait_time)
        return True
    return False

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    file_exists = os.path.isfile(METADATA_FILE)
    with open(METADATA_FILE, mode="a", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["owner", "repo", "description", "language", "size_kb", "stargazers_count", "created_at"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(metadata)

def fetch_repo_metadata(owner, repo):
    url = f"{API_URL}/repos/{owner}/{repo}"
    response = requests.get(url, headers=headers)
    if response.status_code == 403:
        logging.info("Rate limit hit during metadata fetch, rotating token...")
        handle_rate_limit(current_token_index)
        rotate_token()
        return fetch_repo_metadata(owner, repo)
    if response.status_code == 404:
        logging.warning(f"Repo {owner}/{repo} not found for metadata.")
        return None
    if response.status_code == 200:
        return response.json()
    return None

def fetch_repo_files(owner, repo):
    metadata = fetch_repo_metadata(owner, repo)
    if not metadata:
        return {}

    default_branch = metadata.get("default_branch", "main")

    url = f"{API_URL}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
    response = requests.get(url, headers=headers)
    if response.status_code == 403:
        logging.info("Rate limit hit during repo fetch, rotating token...")
        handle_rate_limit(current_token_index)
        rotate_token()
        return fetch_repo_files(owner, repo)
    if response.status_code == 404:
        logging.warning(f"Repo {owner}/{repo} not found or branch {default_branch} missing.")
        return {}
    return response.json()


def fetch_file_content(owner, repo, file_path):
    url = f"{API_URL}/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(url, headers=headers)
    if response.status_code == 403:
        logging.info("Rate limit hit during file fetch, rotating token...")
        handle_rate_limit(current_token_index)
        rotate_token()
        return fetch_file_content(owner, repo, file_path)
    if response.status_code == 404:
        logging.warning(f"File {file_path} not found.")
        return None
    if response.status_code == 200:
        file_data = response.json()
        if 'content' in file_data:
            return base64.b64decode(file_data['content']).decode('utf-8')
    return None

def save_file(owner, repo, file_path, content):
    save_path = os.path.join("dataset", owner, repo, file_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"Saved {file_path} to {save_path}")

def should_download(file_path):
    ext = os.path.splitext(file_path)[1]
    base = os.path.basename(file_path)
    return (
        ext in ALLOWED_EXTENSIONS and
        base not in EXCLUDED_FILENAMES and
        not any(file_path.startswith(folder) for folder in IGNORED_FOLDERS)
    )

def filter_repo(repo_data):
    if "language" in repo_data and repo_data["language"] in allowed_extensions:
        if repo_data["size"] < 50000 and recent_commit(repo_data):
            return True
    return False

def process_repo(owner, repo):
    progress = load_progress()
    fetched_files = progress.get(f"{owner}/{repo}", [])
    
    repo_data = fetch_repo_files(owner, repo)
    if not repo_data:
        return
    
    tree = repo_data.get("tree", [])
    files_to_fetch = [
        file["path"] for file in tree
        if file["type"] == "blob" and should_download(file["path"])
    ]

    for file_path in files_to_fetch:
        if file_path not in fetched_files:
            try:
                content = fetch_file_content(owner, repo, file_path)
                if content:
                    save_file(owner, repo, file_path, content)
                    fetched_files.append(file_path)
            except Exception as e:
                logging.error(f"Error fetching {file_path}: {e}")

            save_progress({f"{owner}/{repo}": fetched_files})
    
    repo_metadata = fetch_repo_metadata(owner, repo)
    if repo_metadata:
        metadata = {
            "owner": owner,
            "repo": repo,
            "description": repo_metadata.get("description", ""),
            "language": repo_metadata.get("language", ""),
            "size_kb": repo_metadata.get("size", 0),
            "stargazers_count": repo_metadata.get("stargazers_count", 0),
            "created_at": repo_metadata.get("created_at", ""),
        }
        save_metadata(metadata)

def collect_data(users_repos):
    for owner, repos in users_repos.items():
        for repo in repos:
            logging.info(f"Processing repository: {repo}")
            process_repo(owner, repo)

# Next to be downloaded
users_repos = {
    "thecrusader25225": ["melody-flow"],
}

if __name__ == "__main__":
    collect_data(users_repos)
