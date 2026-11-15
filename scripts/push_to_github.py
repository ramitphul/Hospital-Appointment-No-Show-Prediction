"""
GitHub push automation script for Hospital No-Show Prediction platform.
Supports both Dulwich (pure Python git) and system git CLI.

Usage:
    python scripts/push_to_github.py --repo https://github.com/<username>/hospital-no-show-prediction.git
    python scripts/push_to_github.py --repo https://github.com/<username>/hospital-no-show-prediction.git --token <github_pat>
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from dulwich import porcelain
from dulwich.repo import Repo


def push_to_remote(repo_url: str, token: str = None, branch: str = "main"):
    """Push local commits to target GitHub repository."""
    repo = Repo(str(root_dir))
    print(f"Opening local Git repository at: {root_dir}")

    # Format authenticated URL if token provided
    target_url = repo_url
    if token:
        if repo_url.startswith("https://"):
            # Format: https://<token>@github.com/user/repo.git
            clean_url = repo_url.replace("https://", "")
            target_url = f"https://{token}@{clean_url}"

    # Set or update origin remote
    config = repo.get_config()
    section = (b"remote", b"origin")
    config.set(section, b"url", target_url.encode("utf-8"))
    config.write_to_path()
    print(f"Configured remote 'origin' -> {repo_url}")

    print(f"Pushing branch '{branch}' to {repo_url}...")
    try:
        porcelain.push(repo, target_url, f"refs/heads/{branch}:refs/heads/{branch}")
        print("\n Successfully pushed project to GitHub!")
        print(f"Repository URL: {repo_url}\n")
    except Exception as e:
        print(f"\nPush encountered error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the remote repository has been created on GitHub.")
        print("2. If the repository is private or requires authentication, provide your GitHub Personal Access Token (PAT):")
        print("   python scripts/push_to_github.py --repo <url> --token <your_personal_access_token>")
        print("3. Or push directly via your terminal if Git is installed:")
        print("   git remote add origin <url>")
        print("   git push -u origin main\n")


def main():
    parser = argparse.ArgumentParser(description="Push local project to GitHub repository.")
    parser.add_argument("--repo", type=str, required=True, help="Target GitHub repository URL (HTTPS or SSH)")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token for authentication")
    parser.add_argument("--branch", type=str, default="main", help="Branch to push (default: main)")
    args = parser.parse_args()

    push_to_remote(repo_url=args.repo, token=args.token, branch=args.branch)


if __name__ == "__main__":
    main()
