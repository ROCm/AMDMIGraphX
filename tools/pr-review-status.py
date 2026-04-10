#!/usr/bin/env python3
"""
Categorize open, non-draft PRs in ROCm/AMDMIGraphX by the number of
reviews from repository members who have write (push) access.

Requirements:
    pip install requests

Usage:
    export GITHUB_TOKEN="ghp_..."
    python pr_review_status.py                  # terminal output
    python pr_review_status.py --json           # JSON to stdout (for dashboard)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import requests

OWNER = "ROCm"
REPO = "AMDMIGraphX"
API = "https://api.github.com"


def get_session() -> requests.Session:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: Set the GITHUB_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return s


# ── helpers ──────────────────────────────────────────────────────────
def paginate(session: requests.Session, url: str, params: Optional[dict] = None):
    """Yield every item across all pages of a paginated GitHub response."""
    params = dict(params or {})
    params.setdefault("per_page", 100)
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        yield from resp.json()
        url = resp.links.get("next", {}).get("url")
        params = {}


def get_members_with_write_access(session: requests.Session) -> set[str]:
    """
    Return the set of GitHub logins that have *push* (write) permission
    or higher on the repository.
    """
    url = f"{API}/repos/{OWNER}/{REPO}/collaborators"
    members: set[str] = set()
    for collab in paginate(session, url, {"permission": "push"}):
        login = collab["login"]
        perms = collab.get("permissions", {})
        if perms.get("push"):
            members.add(login)
    return members


# ── core logic ───────────────────────────────────────────────────────
def get_open_nondraft_prs(session: requests.Session) -> list[dict]:
    """Return every open, non-draft pull request."""
    url = f"{API}/repos/{OWNER}/{REPO}/pulls"
    prs = []
    for pr in paginate(session, url, {"state": "open"}):
        if not pr.get("draft", False):
            prs.append(pr)
    return prs


def get_member_review_count(
    session: requests.Session,
    pr_number: int,
    pr_author: str,
    members: set[str],
) -> tuple[int, list[str], int, list[str]]:
    """
    Count how many distinct repo members have submitted a meaningful
    review (APPROVED or CHANGES_REQUESTED) on the given PR.

    Returns (review_count, reviewers, approval_count, approvers).

    - Only the *latest* review state per reviewer is considered.
    - The PR author's own reviews are excluded.
    """
    url = f"{API}/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews"

    latest: dict[str, str] = {}
    for review in paginate(session, url):
        user = review.get("user", {}).get("login", "")
        state = review.get("state", "")
        if user and state in (
            "APPROVED",
            "CHANGES_REQUESTED",
            "COMMENTED",
            "DISMISSED",
        ):
            latest[user] = state

    member_reviewers = []
    member_approvers = []
    for user, state in latest.items():
        if user == pr_author:
            continue
        if state not in ("APPROVED", "CHANGES_REQUESTED"):
            continue
        if user in members:
            member_reviewers.append(user)
            if state == "APPROVED":
                member_approvers.append(user)

    return len(member_reviewers), member_reviewers, len(member_approvers), member_approvers


def gather_data(session: requests.Session) -> dict:
    """Collect all data and return a structured dict."""
    print("Fetching collaborators with write access…", file=sys.stderr)
    members = get_members_with_write_access(session)
    print(f"  Found {len(members)} members with push access.", file=sys.stderr)

    print("Fetching open, non-draft PRs…", file=sys.stderr)
    prs = get_open_nondraft_prs(session)
    print(f"  Found {len(prs)} open non-draft PRs.", file=sys.stderr)

    buckets: dict[str, list[dict]] = {
        "zero": [],
        "one": [],
        "two_plus": [],
        "two_plus_approved": [],
    }

    for i, pr in enumerate(prs, 1):
        number = pr["number"]
        title = pr["title"]
        author = pr["user"]["login"]
        author_avatar = pr["user"]["avatar_url"]
        html_url = pr["html_url"]
        created_at = pr.get("created_at", "")
        updated_at = pr.get("updated_at", "")
        labels = [l["name"] for l in pr.get("labels", [])]

        member_review_count, reviewers, approval_count, approvers = (
            get_member_review_count(session, number, author, members)
        )

        entry = {
            "number": number,
            "title": title,
            "author": author,
            "author_avatar": author_avatar,
            "url": html_url,
            "created_at": created_at,
            "updated_at": updated_at,
            "labels": labels,
            "member_reviews": member_review_count,
            "reviewers": reviewers,
            "member_approvals": approval_count,
            "approvers": approvers,
        }

        if member_review_count == 0:
            buckets["zero"].append(entry)
        elif member_review_count == 1:
            buckets["one"].append(entry)
        elif approval_count >= 2:
            buckets["two_plus_approved"].append(entry)
        else:
            buckets["two_plus"].append(entry)

        print(
            f"  [{i}/{len(prs)}] PR #{number}: "
            f"{member_review_count} member review(s), "
            f"{approval_count} approval(s)",
            file=sys.stderr,
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": f"{OWNER}/{REPO}",
        "total_prs": len(prs),
        "counts": {
            "zero": len(buckets["zero"]),
            "one": len(buckets["one"]),
            "two_plus": len(buckets["two_plus"]),
            "two_plus_approved": len(buckets["two_plus_approved"]),
        },
        "buckets": buckets,
    }


# ── output formats ───────────────────────────────────────────────────
def print_terminal(data: dict):
    divider = "=" * 80
    labels = {
        "zero": "🔴  Zero member reviews",
        "one": "🟡  One member review",
        "two_plus": "🟢  Two or more member reviews",
        "two_plus_approved": "✅  Two or more member approvals",
    }
    for key in ("zero", "one", "two_plus", "two_plus_approved"):
        items = data["buckets"][key]
        print(f"\n{divider}")
        print(f" {labels[key]}  ({len(items)} PRs)")
        print(divider)
        if not items:
            print("  (none)")
        for pr in items:
            reviewers = ", ".join(pr["reviewers"]) if pr["reviewers"] else "—"
            print(f"  #{pr['number']:>5}  [{pr['author']}]  {pr['title']}")
            print(f"         Reviewers: {reviewers}")
            print(f"         {pr['url']}")

    print(f"\n{divider}")
    print(" Summary")
    print(divider)
    print(f"  Total open non-draft PRs : {data['total_prs']}")
    print(f"  Zero member reviews      : {data['counts']['zero']}")
    print(f"  One member review        : {data['counts']['one']}")
    print(f"  Two+ member reviews      : {data['counts']['two_plus']}")
    print(f"  Two+ member approvals    : {data['counts']['two_plus_approved']}")


def main():
    parser = argparse.ArgumentParser(
        description="PR review status report for ROCm/AMDMIGraphX"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON to stdout (for the GitHub Pages dashboard)",
    )
    args = parser.parse_args()

    session = get_session()
    data = gather_data(session)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_terminal(data)


if __name__ == "__main__":
    main()
