import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

def upload_file_to_huggingface(local_path: str, path_in_repo: str):
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID")

    if not token or not repo_id:
        return {
            "uploaded": False,
            "message": "HF_TOKEN or HF_REPO_ID is missing. Add them to .env file."
        }

    api = HfApi(token=token)

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        exist_ok=True,
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
    }
