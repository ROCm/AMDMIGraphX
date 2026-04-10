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
import html
import json
import os
import re
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


# ── GraphQL ───────────────────────────────────────────────────────────
GRAPHQL_URL = "https://api.github.com/graphql"

PR_QUERY = """
query($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(states: OPEN, first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        isDraft
        url
        createdAt
        updatedAt
        author { login avatarUrl }
        labels(first: 20) { nodes { name } }
        reviews(first: 100) {
          nodes { author { login } state }
        }
        commits(last: 1) {
          nodes {
            commit {
              statusCheckRollup { state }
            }
          }
        }
        comments(last: 100) {
          nodes { author { login } body updatedAt }
        }
      }
    }
  }
}
"""


def graphql_query(session: requests.Session, query: str, variables: dict) -> dict:
    resp = session.post(GRAPHQL_URL, json={"query": query, "variables": variables})
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL errors: {body['errors']}")
    return body["data"]


def fetch_all_prs(session: requests.Session) -> list[dict]:
    """Fetch all open, non-draft PRs with reviews, CI, and comments via GraphQL."""
    all_prs: list[dict] = []
    cursor = None
    while True:
        data = graphql_query(session, PR_QUERY, {
            "owner": OWNER, "repo": REPO, "cursor": cursor,
        })
        connection = data["repository"]["pullRequests"]
        for node in connection["nodes"]:
            if not node["isDraft"]:
                all_prs.append(node)
        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
    return all_prs


# ── processing helpers ────────────────────────────────────────────────
DASH_NOTE_RE = re.compile(r"^#dash_note\s+(.*)", re.DOTALL)


def sanitize_note(raw: str) -> str:
    """Escape HTML, collapse whitespace, and truncate."""
    text = html.escape(raw.strip())
    text = re.sub(r"\s+", " ", text)
    return text[:500]


def extract_reviews(
    review_nodes: list[dict],
    pr_author: str,
    members: set[str],
) -> tuple[int, list[str], int, list[str]]:
    """Process GraphQL review nodes into (review_count, reviewers, approval_count, approvers)."""
    latest: dict[str, str] = {}
    for review in review_nodes:
        author = (review.get("author") or {}).get("login", "")
        state = review.get("state", "")
        if author and state in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED"):
            latest[author] = state

    member_reviewers = []
    member_approvers = []
    for user, state in latest.items():
        if user == pr_author:
            continue
        if state not in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED"):
            continue
        if user in members:
            member_reviewers.append(user)
            if state == "APPROVED":
                member_approvers.append(user)

    return len(member_reviewers), member_reviewers, len(member_approvers), member_approvers


def extract_ci_status(commits_nodes: list[dict]) -> str:
    """Map statusCheckRollup.state to our status string."""
    if not commits_nodes:
        return "none"
    rollup = (
        commits_nodes[0]
        .get("commit", {})
        .get("statusCheckRollup")
    )
    if rollup is None:
        return "none"
    state = rollup.get("state", "").upper()
    return {
        "SUCCESS": "success",
        "FAILURE": "failure",
        "ERROR": "failure",
        "PENDING": "pending",
        "EXPECTED": "pending",
    }.get(state, "none")


def extract_dash_note(
    comment_nodes: list[dict],
    members: set[str],
) -> Optional[dict]:
    """Find the most recent #dash_note from a member (comments are in chronological order)."""
    best: Optional[dict] = None
    for comment in comment_nodes:
        user = (comment.get("author") or {}).get("login", "")
        if user not in members:
            continue
        body = (comment.get("body") or "").strip()
        m = DASH_NOTE_RE.match(body)
        if not m:
            continue
        best = {
            "author": user,
            "body": sanitize_note(m.group(1)),
            "updated_at": comment.get("updatedAt", ""),
        }
    return best


# ── core logic ───────────────────────────────────────────────────────
def gather_data(session: requests.Session) -> dict:
    """Collect all data and return a structured dict."""
    print("Fetching collaborators with write access…", file=sys.stderr)
    members = get_members_with_write_access(session)
    print(f"  Found {len(members)} members with push access.", file=sys.stderr)

    print("Fetching open, non-draft PRs (GraphQL)…", file=sys.stderr)
    prs = fetch_all_prs(session)
    print(f"  Found {len(prs)} open non-draft PRs.", file=sys.stderr)

    buckets: dict[str, list[dict]] = {
        "needs_reviews": [],
        "in_review": [],
        "approved": [],
        "ready_to_merge": [],
    }

    for i, pr in enumerate(prs, 1):
        number = pr["number"]
        author = (pr.get("author") or {}).get("login", "")
        author_avatar = (pr.get("author") or {}).get("avatarUrl", "")

        review_nodes = pr.get("reviews", {}).get("nodes", [])
        member_review_count, reviewers, approval_count, approvers = (
            extract_reviews(review_nodes, author, members)
        )

        ci_status = extract_ci_status(
            pr.get("commits", {}).get("nodes", [])
        )

        dash_note = extract_dash_note(
            pr.get("comments", {}).get("nodes", []), members
        )

        entry = {
            "number": number,
            "title": pr["title"],
            "author": author,
            "author_avatar": author_avatar,
            "url": pr["url"],
            "created_at": pr.get("createdAt", ""),
            "updated_at": pr.get("updatedAt", ""),
            "labels": [n["name"] for n in pr.get("labels", {}).get("nodes", [])],
            "member_reviews": member_review_count,
            "reviewers": reviewers,
            "member_approvals": approval_count,
            "approvers": approvers,
            "ci_status": ci_status,
            "dash_note": dash_note,
        }

        if approval_count >= 2 and ci_status == "success":
            buckets["ready_to_merge"].append(entry)
        elif approval_count >= 2:
            buckets["approved"].append(entry)
        elif member_review_count >= 2:
            buckets["in_review"].append(entry)
        else:
            buckets["needs_reviews"].append(entry)

        print(
            f"  [{i}/{len(prs)}] PR #{number}: "
            f"{member_review_count} review(s), "
            f"{approval_count} approval(s), "
            f"CI={ci_status}",
            file=sys.stderr,
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": f"{OWNER}/{REPO}",
        "total_prs": len(prs),
        "counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
    }


# ── output formats ───────────────────────────────────────────────────
CI_ICONS = {"success": "✅", "failure": "❌", "pending": "🟠", "none": "⚪"}

def print_terminal(data: dict):
    divider = "=" * 80
    labels = {
        "needs_reviews":  "🔴  Needs Reviews",
        "in_review":      "🟡  Has 2+ Reviewers",
        "approved":       "🟣  Has 2+ Approvals",
        "ready_to_merge": "🟢  Ready to Merge",
    }
    for key in ("needs_reviews", "in_review", "approved", "ready_to_merge"):
        items = data["buckets"][key]
        print(f"\n{divider}")
        print(f" {labels[key]}  ({len(items)} PRs)")
        print(divider)
        if not items:
            print("  (none)")
        for pr in items:
            ci = CI_ICONS.get(pr.get("ci_status", "none"), "⚪")
            approvals = pr.get("member_approvals", 0)
            reviewers = ", ".join(pr["reviewers"]) if pr["reviewers"] else "—"
            print(f"  #{pr['number']:>5}  [{pr['author']}]  {pr['title']}")
            print(f"         Approvals: {approvals}/2  CI: {ci}  Reviewers: {reviewers}")
            if pr.get("dash_note"):
                note = pr["dash_note"]
                print(f"         Note ({note['author']}): {note['body']}")
            print(f"         {pr['url']}")

    print(f"\n{divider}")
    print(" Summary")
    print(divider)
    print(f"  Total open non-draft PRs : {data['total_prs']}")
    for key in ("needs_reviews", "in_review", "approved", "ready_to_merge"):
        print(f"  {labels[key]:30s}: {data['counts'][key]}")


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
