"""
GitHub push automation script for Hospital No-Show Prediction platform.
Supports both Dulwich (pure Python git) and system git CLI.

Usage:
    python scripts/push_to_github.py --repo https://github.com/ramitphul/Hospital-Appointment-No-Show-Prediction.git
    python scripts/push_to_github.py --repo https://github.com/ramitphul/Hospital-Appointment-No-Show-Prediction.git --token <github_pat>
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from dulwich import porcelain
from dulwich.repo import Repo


def push_to_remote(repo_url: str, token: str = None, username: str = None, branch: str = "main"):
    """Push local commits to target GitHub repository."""
    repo = Repo(str(root_dir))
    print(f"Opening local Git repository at: {root_dir}")

    # Fallback to env var if token not provided via flag
    if not token:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    # Parse repo owner if username not provided
    if not username:
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            username = path_parts[0]
        else:
            username = "git"

    # Set or update origin remote in git config
    config = repo.get_config()
    section = (b"remote", b"origin")
    config.set(section, b"url", repo_url.encode("utf-8"))
    config.write_to_path()
    print(f"Configured remote 'origin' -> {repo_url}")

    # 1. Try standard git CLI if installed
    git_bin = shutil.which("git")
    if git_bin:
        print(f"Detected system git at: {git_bin}. Attempting git push...")
        auth_url = repo_url
        if token and repo_url.startswith("https://"):
            auth_url = repo_url.replace("https://", f"https://{username}:{token}@")
        res = subprocess.run([git_bin, "push", "-u", auth_url, branch], cwd=str(root_dir))
        if res.returncode == 0:
            print("\n Successfully pushed project to GitHub via git CLI!")
            print(f"Repository URL: {repo_url}\n")
            return

    # 2. Use Dulwich pure-Python push
    print(f"Pushing branch '{branch}' to {repo_url} via Python Git client...")
    kwargs = {}
    target_location = repo_url
    if token:
        kwargs["username"] = username
        kwargs["password"] = token
        if repo_url.startswith("https://"):
            target_location = repo_url.replace("https://", f"https://{username}:{token}@")

    try:
        porcelain.push(
            repo=repo,
            remote_location=target_location,
            refspecs=f"refs/heads/{branch}:refs/heads/{branch}".encode("utf-8"),
            **kwargs
        )
        print("\n Successfully pushed project to GitHub!")
        print(f"Repository URL: {repo_url}\n")
    except Exception as e:
        print(f"\nPush encountered error: {e}")
        print("\nTroubleshooting & Next Steps:")
        print("1. GitHub requires authentication (Personal Access Token) to push.")
        print("   Create a fine-grained or classic token with 'repo' scope at:")
        print("   https://github.com/settings/tokens")
        print("2. Re-run this push script with your token:")
        print(f"   python scripts/push_to_github.py --repo {repo_url} --token <YOUR_GITHUB_TOKEN>")
        print("3. Or set the environment variable:")
        print(f"   $env:GITHUB_TOKEN = '<YOUR_GITHUB_TOKEN>'")
        print(f"   python scripts/push_to_github.py --repo {repo_url}\n")


def main():
    parser = argparse.ArgumentParser(description="Push local project to GitHub repository.")
    parser.add_argument("--repo", type=str, required=True, help="Target GitHub repository URL (HTTPS or SSH)")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token for authentication")
    parser.add_argument("--username", type=str, default=None, help="GitHub username")
    parser.add_argument("--branch", type=str, default="main", help="Branch to push (default: main)")
    args = parser.parse_args()

    push_to_remote(repo_url=args.repo, token=args.token, username=args.username, branch=args.branch)


if __name__ == "__main__":
    main()
