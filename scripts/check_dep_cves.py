#!/usr/bin/env python3
"""
Scan dependency minimums in pyproject.toml and requirements*.txt for known CVEs
using the OSV (Open Source Vulnerability) database.

Works as:
  - a pre-commit hook (triggered on pyproject.toml / requirements file changes)
  - a GitHub Actions step (posts PR comment when CVEs are found)
  - a local CLI for ad-hoc auditing

Exit codes:
  0  all versions clean (or network unavailable in non-strict mode)
  1  one or more vulnerable minimum versions found
  2  usage / parse error

Usage:
  python3 scripts/check_dep_cves.py                    # scan pyproject.toml + requirements*.txt
  python3 scripts/check_dep_cves.py --fix              # suggest safe minimum versions
  python3 scripts/check_dep_cves.py --strict           # fail on network error instead of warn
  python3 scripts/check_dep_cves.py --json             # emit JSON for GH Actions annotation
  python3 scripts/check_dep_cves.py requirements.txt   # scan a specific file
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ECOSYSTEM = "PyPI"
OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{id}"

# Matches >=1.0, ==1.0, ~=1.0, ===1.0  (captures the version string after the operator)
_GE_RE = re.compile(r"(?:>=|==|~=|===)\s*(\d[\w.*+\-!]*)")
# Bare pinned version with no operator (e.g. "1.2.3" in a requirements.txt)
_BARE_RE = re.compile(r"^\d[\w.]*$")


# ── Version parsing ────────────────────────────────────────────────────────────


def extract_min_version(specifier: str) -> str | None:
    """Return the effective lower-bound version from a PEP 508 specifier string."""
    specifier = specifier.strip()
    if not specifier:
        return None
    if _BARE_RE.match(specifier):
        return specifier
    m = _GE_RE.search(specifier)
    return m.group(1).rstrip(".*") if m else None


def _parse_dep(raw: str, source: str) -> tuple[str, str, str] | None:
    """Parse one dependency string → (normalised_name, min_version, source) or None."""
    # Drop environment markers and inline comments
    dep = raw.split(";")[0].split("#")[0].strip()
    if not dep or dep.startswith("-"):
        return None
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)", dep)
    if not m:
        return None
    name = _normalise(m.group(1))
    min_ver = extract_min_version(m.group(2).strip())
    return (name, min_ver, source) if min_ver else None


def _normalise(name: str) -> str:
    """Normalise PyPI package name (PEP 503): lowercase, collapse [-_.] to -."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ── File parsers ───────────────────────────────────────────────────────────────


def parse_pyproject(path: Path) -> list[tuple[str, str, str]]:
    """Return [(name, min_ver, group_label), ...] from pyproject.toml."""
    import tomllib  # stdlib ≥ 3.11

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    entries: list[tuple[str, str, str]] = []
    project = data.get("project", {})

    for dep in project.get("dependencies", []):
        r = _parse_dep(dep, "core")
        if r:
            entries.append(r)

    for group, deps in project.get("optional-dependencies", {}).items():
        for dep in deps:
            r = _parse_dep(dep, group)
            if r:
                entries.append(r)

    return entries


def parse_requirements(path: Path) -> list[tuple[str, str, str]]:
    """Return [(name, min_ver, filename), ...] from a requirements*.txt file."""
    entries: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = _parse_dep(line, path.name)
        if r:
            entries.append(r)
    return entries


# ── OSV queries ────────────────────────────────────────────────────────────────


def query_osv_batch(
    packages: list[tuple[str, str]],
    strict: bool,
) -> list[dict] | None:
    """
    Batch-query OSV for (name, version) pairs.
    Returns a list of result dicts in the same order, or None on network failure.
    """
    if not packages:
        return []
    payload = json.dumps(
        {
            "queries": [
                {"version": ver, "package": {"name": pkg, "ecosystem": ECOSYSTEM}}
                for pkg, ver in packages
            ]
        }
    ).encode()
    req = urllib.request.Request(
        OSV_BATCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        # S310: URL is the module-level constant OSV_BATCH_URL (https://), not user input.
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read())["results"]
    except urllib.error.URLError as exc:
        msg = f"OSV API unreachable: {exc}"
        if strict:
            print(f"❌ {msg}", file=sys.stderr)
            return None
        print(
            f"⚠️  {msg} — skipping CVE check (use --strict to fail on network errors).",
            file=sys.stderr,
        )
        return [{}] * len(packages)


def fetch_fixed_version(pkg: str, vuln_ids: list[str]) -> str | None:
    """
    Inspect OSV vuln records to find the lowest version that is fixed.
    Samples at most 5 IDs to stay fast.
    """
    fixed: list[str] = []
    for vid in vuln_ids[:5]:
        try:
            url = OSV_VULN_URL.format(id=vid)
            # S310: URL built from OSV_VULN_URL constant (https://), not user input.
            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
                vuln = json.loads(resp.read())
            for aff in vuln.get("affected", []):
                if _normalise(aff.get("package", {}).get("name", "")) == _normalise(pkg):
                    for rng in aff.get("ranges", []):
                        for ev in rng.get("events", []):
                            if "fixed" in ev and re.match(r"^\d", ev["fixed"]):
                                fixed.append(ev["fixed"])
        except Exception as exc:
            print(f"⚠️  Could not fetch fix version for {vid}: {exc}", file=sys.stderr)

    if not fixed:
        return None
    try:
        from packaging.version import Version  # type: ignore[import-untyped]  # optional fast path

        return str(max(fixed, key=Version))
    except ImportError:
        return sorted(fixed, key=lambda v: [int(x) for x in re.findall(r"\d+", v)])[-1]


# ── Reporting ──────────────────────────────────────────────────────────────────


def _make_summary(vulnerable: list[dict], *, suggest_fix: bool) -> str:
    """Build the human-readable / Markdown summary block."""
    lines = ["## Dependency CVE Check — Vulnerable Minimum Versions Found\n"]
    lines.append(f"**{len(vulnerable)} package(s)** pin a minimum version with known CVEs.\n")
    header = "| Package | Min version | Source | CVE count | CVE IDs |"
    header += " Suggested fix |" if suggest_fix else ""
    lines.append(header)
    sep = "|---|---|---|---|---|"
    sep += "---|" if suggest_fix else ""
    lines.append(sep)
    for v in vulnerable:
        ids_cell = ", ".join(v["ids"][:5]) + ("…" if len(v["ids"]) > 5 else "")
        row = f"| `{v['name']}` | `{v['min_ver']}` | {v['source']} | {len(v['ids'])} | {ids_cell} |"
        if suggest_fix:
            fix_cell = v.get("fix", "—")
            row += f" `{fix_cell}` |"
        lines.append(row)
    lines.append(
        "\n> Bump the minimum version in `pyproject.toml` (or requirements file) to the "
        "suggested fix, then re-run `python3 scripts/check_dep_cves.py --fix` to verify."
    )
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    args = set(argv[1:])
    suggest_fix = "--fix" in args
    strict = "--strict" in args
    emit_json = "--json" in args
    positional = [a for a in argv[1:] if not a.startswith("--")]

    if positional:
        paths = [Path(p) for p in positional]
    else:
        paths = [Path("pyproject.toml")]
        paths += sorted(Path(".").glob("requirements*.txt"))

    entries: list[tuple[str, str, str]] = []
    for p in paths:
        if not p.exists():
            print(f"⚠️  {p} not found — skipping.", file=sys.stderr)
            continue
        try:
            if p.name == "pyproject.toml":
                entries.extend(parse_pyproject(p))
            else:
                entries.extend(parse_requirements(p))
        except Exception as exc:
            print(f"❌ Failed to parse {p}: {exc}", file=sys.stderr)
            return 2

    if not entries:
        print("No pinned dependency minimums found — nothing to check.")
        return 0

    print(f"Checking {len(entries)} dependency minimum(s) against OSV …", file=sys.stderr)

    pkg_ver_pairs = [(name, ver) for name, ver, _ in entries]
    results = query_osv_batch(pkg_ver_pairs, strict=strict)
    if results is None:
        return 1  # strict mode + network failure

    # Collect vulnerable entries
    vulnerable: list[dict] = []
    for (name, min_ver, source), res in zip(entries, results, strict=True):
        vuln_ids = [v["id"] for v in res.get("vulns", [])]
        if not vuln_ids:
            continue
        rec: dict = {"name": name, "min_ver": min_ver, "source": source, "ids": vuln_ids}
        if suggest_fix:
            rec["fix"] = fetch_fixed_version(name, vuln_ids) or "check PyPI"
        vulnerable.append(rec)

    # Pretty-print table to stdout
    col_name = max((len(e[0]) for e in entries), default=7)
    col_ver = max((len(e[1]) for e in entries), default=11)
    col_src = max((len(e[2]) for e in entries), default=6)
    header = f"{'Package':<{col_name}}  {'Min version':<{col_ver}}  {'Source':<{col_src}}  Status"
    print(header)
    print("─" * len(header))

    vuln_names = {v["name"] for v in vulnerable}
    for name, min_ver, source in entries:
        if name in vuln_names:
            rec = next(v for v in vulnerable if v["name"] == name)
            ids_short = ", ".join(rec["ids"][:3]) + ("…" if len(rec["ids"]) > 3 else "")
            status = f"⚠️  {len(rec['ids'])} CVE(s): {ids_short}"
            if suggest_fix and rec.get("fix"):
                status += f"  →  fix: >={rec['fix']}"
        else:
            status = "✅ clean"
        print(f"{name:<{col_name}}  {min_ver:<{col_ver}}  {source:<{col_src}}  {status}")

    if not vulnerable:
        print("\n✅ All dependency minimums are clean.")
        if emit_json:
            print(json.dumps({"status": "clean", "vulnerable": []}))
        return 0

    summary = _make_summary(vulnerable, suggest_fix=suggest_fix)
    print(f"\n{summary}")

    if emit_json:
        print(json.dumps({"status": "vulnerable", "vulnerable": vulnerable}))

    if not suggest_fix:
        print("\nTip: run with --fix to get suggested safe minimum versions.", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
