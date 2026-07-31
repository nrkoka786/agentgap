#!/usr/bin/env python3
"""
AgentGap Scanner — Basic GitHub Repository Discovery

What it does:
1. Searches GitHub across several AI categories.
2. Deduplicates repositories found in multiple searches.
3. Applies a lightweight AI relevance filter.
4. Computes transparent popularity, activity, and strategic scores.
5. Exports CSV and JSON files.

This first version does not scan Issues or Discussions yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

API_URL = "https://api.github.com/search/repositories"
API_VERSION = "2026-03-10"

AI_TERMS = {
    "ai", "artificial intelligence", "llm", "large language model", "agent",
    "agents", "rag", "retrieval augmented", "mcp", "model context protocol",
    "machine learning", "deep learning", "transformer", "generative ai",
    "copilot", "inference", "embedding", "vector database", "multimodal",
    "computer vision", "speech recognition", "text to speech"
}

NOISE_TERMS = {
    "awesome list", "awesome-", "roadmap", "interview questions", "cheatsheet",
    "cheat sheet", "course", "tutorial collection", "learning resources"
}


@dataclass
class RepoRecord:
    full_name: str
    html_url: str
    owner: str
    name: str
    description: str
    stars: int
    forks: int
    watchers: int
    open_issues: int
    language: str
    license: str
    topics: list[str]
    created_at: str
    updated_at: str
    pushed_at: str
    default_branch: str
    is_fork: bool
    archived: bool
    disabled: bool
    visibility: str
    categories: set[str] = field(default_factory=set)
    matching_queries: set[str] = field(default_factory=set)
    relevance_score: float = 0.0
    popularity_score: float = 0.0
    activity_score: float = 0.0
    strategic_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["categories"] = sorted(self.categories)
        d["matching_queries"] = sorted(self.matching_queries)
        d["topics"] = sorted(self.topics)
        return d


class GitHubScanner:
    def __init__(self, token: str, delay_seconds: float = 2.2) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "AgentGap-Scanner/0.1",
        })
        self.delay_seconds = delay_seconds

    def search(self, query: str, pages: int, per_page: int = 100) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(per_page, 100),
                "page": page,
            }
            response = self.session.get(API_URL, params=params, timeout=45)

            if response.status_code == 403:
                reset = response.headers.get("X-RateLimit-Reset")
                message = response.json().get("message", "GitHub rate limit or access error.")
                if reset and reset.isdigit():
                    reset_dt = datetime.fromtimestamp(int(reset), tz=timezone.utc)
                    raise RuntimeError(f"{message} Rate limit resets at {reset_dt.isoformat()}.")
                raise RuntimeError(message)

            if response.status_code == 422:
                raise RuntimeError(
                    f"GitHub rejected this search query:\n{query}\n"
                    f"Response: {response.text[:500]}"
                )

            response.raise_for_status()
            payload = response.json()
            page_items = payload.get("items", [])
            results.extend(page_items)

            print(f"    page {page}: {len(page_items)} repositories")
            if len(page_items) < per_page:
                break
            time.sleep(self.delay_seconds)
        return results


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def days_since(value: str) -> int:
    if not value:
        return 9999
    delta = datetime.now(timezone.utc) - parse_iso(value)
    return max(delta.days, 0)


def contains_any(text: str, terms: set[str]) -> int:
    low = text.lower()
    return sum(1 for term in terms if term in low)


def calculate_relevance(repo: RepoRecord) -> float:
    text = " ".join([
        repo.name,
        repo.description,
        " ".join(repo.topics),
        " ".join(repo.categories),
    ]).lower()

    positive = contains_any(text, AI_TERMS)
    negative = contains_any(text, NOISE_TERMS)

    score = min(100.0, positive * 18.0)
    score += min(15.0, len(repo.categories) * 3.0)
    score -= negative * 25.0
    if repo.is_fork:
        score -= 30.0
    if repo.archived or repo.disabled:
        score -= 60.0
    return max(0.0, min(100.0, score))


def calculate_scores(repo: RepoRecord, max_stars: int, max_forks: int) -> None:
    star_component = math.log1p(repo.stars) / max(math.log1p(max_stars), 1)
    fork_component = math.log1p(repo.forks) / max(math.log1p(max_forks), 1)
    repo.popularity_score = round(100 * (0.75 * star_component + 0.25 * fork_component), 2)

    pushed_days = days_since(repo.pushed_at)
    updated_days = days_since(repo.updated_at)
    push_activity = math.exp(-pushed_days / 180)
    update_activity = math.exp(-updated_days / 240)
    repo.activity_score = round(100 * (0.7 * push_activity + 0.3 * update_activity), 2)

    repo.strategic_score = round(
        0.40 * repo.popularity_score
        + 0.30 * repo.activity_score
        + 0.30 * repo.relevance_score,
        2,
    )


def convert_item(item: dict[str, Any], category: str, query: str) -> RepoRecord:
    license_obj = item.get("license") or {}
    owner_obj = item.get("owner") or {}
    return RepoRecord(
        full_name=item.get("full_name", ""),
        html_url=item.get("html_url", ""),
        owner=owner_obj.get("login", ""),
        name=item.get("name", ""),
        description=item.get("description") or "",
        stars=int(item.get("stargazers_count", 0)),
        forks=int(item.get("forks_count", 0)),
        watchers=int(item.get("watchers_count", 0)),
        open_issues=int(item.get("open_issues_count", 0)),
        language=item.get("language") or "",
        license=license_obj.get("spdx_id") or "",
        topics=item.get("topics") or [],
        created_at=item.get("created_at") or "",
        updated_at=item.get("updated_at") or "",
        pushed_at=item.get("pushed_at") or "",
        default_branch=item.get("default_branch") or "",
        is_fork=bool(item.get("fork", False)),
        archived=bool(item.get("archived", False)),
        disabled=bool(item.get("disabled", False)),
        visibility=item.get("visibility") or "public",
        categories={category},
        matching_queries={query},
    )


def merge_repo(existing: RepoRecord, item: dict[str, Any], category: str, query: str) -> None:
    existing.categories.add(category)
    existing.matching_queries.add(query)
    # Search results may differ slightly over time. Keep current counts.
    existing.stars = max(existing.stars, int(item.get("stargazers_count", 0)))
    existing.forks = max(existing.forks, int(item.get("forks_count", 0)))
    existing.open_issues = max(existing.open_issues, int(item.get("open_issues_count", 0)))


def write_csv(path: Path, repos: list[RepoRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, repo in enumerate(repos, start=1):
        d = repo.to_dict()
        d["rank"] = rank
        d["topics"] = ", ".join(d["topics"])
        d["categories"] = ", ".join(d["categories"])
        d["matching_queries"] = " | ".join(d["matching_queries"])
        rows.append(d)

    if not rows:
        return

    fieldnames = ["rank"] + [k for k in rows[0].keys()]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, repos: list[RepoRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"rank": rank, **repo.to_dict()}
        for rank, repo in enumerate(repos, start=1)
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover and rank AI repositories on GitHub.")
    parser.add_argument("--queries", default="queries.json", help="Path to search query JSON.")
    parser.add_argument("--pages", type=int, default=2, help="Pages per query; 100 results per page.")
    parser.add_argument("--limit", type=int, default=500, help="Number of ranked repositories to export.")
    parser.add_argument("--min-stars", type=int, default=20, help="Final minimum star count.")
    parser.add_argument("--min-relevance", type=float, default=20, help="Minimum relevance score, 0-100.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token or token.startswith("github_pat_your"):
        print("ERROR: Add your GitHub token to a .env file. See .env.example.", file=sys.stderr)
        return 2

    query_path = Path(args.queries)
    if not query_path.exists():
        print(f"ERROR: Query file not found: {query_path}", file=sys.stderr)
        return 2

    query_defs = json.loads(query_path.read_text(encoding="utf-8"))
    scanner = GitHubScanner(token)
    repo_map: dict[str, RepoRecord] = {}

    for index, item in enumerate(query_defs, start=1):
        category = item["category"]
        query = item["query"]
        print(f"[{index}/{len(query_defs)}] {category}")
        try:
            found = scanner.search(query, pages=max(1, args.pages))
        except Exception as exc:
            print(f"  WARNING: query failed: {exc}", file=sys.stderr)
            continue

        for raw in found:
            full_name = raw.get("full_name", "").lower()
            if not full_name:
                continue
            if full_name in repo_map:
                merge_repo(repo_map[full_name], raw, category, query)
            else:
                repo_map[full_name] = convert_item(raw, category, query)

        time.sleep(scanner.delay_seconds)

    repos = list(repo_map.values())
    for repo in repos:
        repo.relevance_score = round(calculate_relevance(repo), 2)

    repos = [
        r for r in repos
        if not r.is_fork
        and not r.archived
        and not r.disabled
        and r.stars >= args.min_stars
        and r.relevance_score >= args.min_relevance
    ]

    if not repos:
        print("No repositories passed the filters. Try lowering --min-relevance.", file=sys.stderr)
        return 1

    max_stars = max(r.stars for r in repos)
    max_forks = max(r.forks for r in repos)
    for repo in repos:
        calculate_scores(repo, max_stars, max_forks)

    repos.sort(key=lambda r: (r.strategic_score, r.stars), reverse=True)
    repos = repos[: max(1, args.limit)]

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "top_ai_repositories.csv"
    json_path = output_dir / "top_ai_repositories.json"
    write_csv(csv_path, repos)
    write_json(json_path, repos)

    print()
    print(f"Complete: exported {len(repos)} repositories")
    print(f"CSV:  {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    print()
    print("Top 10:")
    for rank, repo in enumerate(repos[:10], start=1):
        print(
            f"{rank:>2}. {repo.full_name:<42} "
            f"score={repo.strategic_score:>6.2f} stars={repo.stars}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
