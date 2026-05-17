import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)


def upload_file_to_huggingface(local_path: str, path_in_repo: str):
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID")

    if not token or not repo_id:
        return {
            "uploaded": False,
            "message": f"HF_TOKEN or HF_REPO_ID is missing. Checked .env at: {ENV_PATH}"
        }

    local_path = str(local_path)

    if not os.path.exists(local_path):
        return {
            "uploaded": False,
            "message": f"Local file not found: {local_path}"
        }

    try:
        api = HfApi(token=token)

        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            exist_ok=True,
            private=False
        )

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        return {
            "uploaded": True,
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "message": "File uploaded to Hugging Face successfully."
        }

    except Exception as e:
        return {
            "uploaded": False,
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "message": f"Upload failed: {str(e)}"
        }