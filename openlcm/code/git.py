"""Git integration for OpenLCM LST — clone remote repos and extract metadata.

Supports GitHub/GitLab/Bitbucket URLs and any git-accessible remote.
Repos are cached in ~/.openlcm/repos/ so repeated scans just pull the latest.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_URL_PATTERNS = [
    re.compile(r"^https?://"),
    re.compile(r"^git@"),
    re.compile(r"^ssh://"),
]

_DEFAULT_CACHE = Path.home() / ".openlcm" / "repos"


@dataclass
class CloneResult:
    local_path: Path
    origin_url: str
    branch: str
    commit_hash: str
    commit_message: str
    cloned_at: float
    was_cached: bool       # True if we reused an existing clone (pulled latest)


def is_url(path_or_url: str) -> bool:
    """Return True if the string looks like a remote git URL."""
    s = path_or_url.strip()
    return any(p.match(s) for p in _URL_PATTERNS)


class GitCloner:
    """Clone and cache remote git repositories."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE)

    def clone(
        self,
        url: str,
        branch: Optional[str] = None,
        depth: int = 1,
        force_refresh: bool = False,
    ) -> CloneResult:
        """
        Clone a remote repo (or pull latest if already cached).

        Args:
            url:           Remote git URL
            branch:        Specific branch/tag (default: remote HEAD)
            depth:         Shallow clone depth (default 1; 0 = full history)
            force_refresh: Pull latest even if already cached

        Returns:
            CloneResult with local_path and metadata
        """
        if not shutil.which("git"):
            raise RuntimeError("git not found in PATH — install git to clone remote repos")

        dest = self._cache_path(url)
        was_cached = dest.exists()

        if was_cached and not force_refresh:
            # Pull latest on the cached clone
            logger.info("Updating cached clone at %s", dest)
            _run(["git", "-C", str(dest), "pull", "--ff-only", "--quiet"], check=False)
        else:
            if was_cached and force_refresh:
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)

            cmd = ["git", "clone", "--quiet"]
            if depth and depth > 0:
                cmd += ["--depth", str(depth)]
            if branch:
                cmd += ["--branch", branch]
            cmd += [url, str(dest)]

            logger.info("Cloning %s → %s", url, dest)
            _run(cmd)

        meta = get_repo_metadata(dest)
        return CloneResult(
            local_path=dest,
            origin_url=url,
            branch=meta.get("branch", ""),
            commit_hash=meta.get("commit_hash", ""),
            commit_message=meta.get("commit_message", ""),
            cloned_at=time.time(),
            was_cached=was_cached,
        )

    def _cache_path(self, url: str) -> Path:
        """Derive a stable cache directory from a remote URL."""
        # Normalize: strip .git suffix, extract host/path
        clean = url.strip()
        clean = re.sub(r"\.git$", "", clean)
        clean = re.sub(r"^https?://", "", clean)
        clean = re.sub(r"^git@([^:]+):", r"\1/", clean)
        clean = re.sub(r"^ssh://[^/]+/", "", clean)
        # Replace path separators with underscores for safety
        slug = re.sub(r"[^\w/\-]", "_", clean).strip("/")
        return self.cache_dir / slug


def get_repo_metadata(repo_path: str | Path) -> dict:
    """
    Extract git metadata from any local git repository.

    Returns dict with: commit_hash, commit_message, commit_date,
                       branch, origin_url, file_count (from git ls-files)
    """
    root = Path(repo_path)
    result: dict = {
        "commit_hash": "",
        "commit_message": "",
        "commit_date": "",
        "branch": "",
        "origin_url": "",
    }

    if not shutil.which("git"):
        return result

    def _git(*args: str) -> str:
        out = _run(["git", "-C", str(root)] + list(args), check=False, capture=True)
        return out.strip() if out else ""

    try:
        log_line = _git("log", "-1", "--format=%H|%s|%ai")
        if log_line and "|" in log_line:
            parts = log_line.split("|", 2)
            result["commit_hash"] = parts[0]
            result["commit_message"] = parts[1]
            result["commit_date"] = parts[2] if len(parts) > 2 else ""

        result["branch"] = (
            _git("branch", "--show-current")
            or _git("rev-parse", "--abbrev-ref", "HEAD")
        )
        result["origin_url"] = _git("remote", "get-url", "origin")
    except Exception as exc:
        logger.debug("git metadata extraction failed: %s", exc)

    return result


# ── Internal ──────────────────────────────────────────────────────────────────

def _run(
    cmd: list[str],
    check: bool = True,
    capture: bool = False,
) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
            )
        return result.stdout if capture else None
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out: {' '.join(cmd)}") from exc
