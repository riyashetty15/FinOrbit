#!/usr/bin/env python3
"""
CLI script for ingesting regulatory documents into the FinOrbit RAG pipeline.

Supports:
  - Batch ingestion from a directory (--dir)
  - Single file ingestion (--file)
  - Per-issuer metadata templates (RBI / SEBI / IRDAI / CBDT)
  - --dry-run mode: validates inputs without calling the API
  - JSON ingestion report (--output-report)

Usage examples:
  # Single file
  python scripts/ingest_regulatory_docs.py \\
      --file RBI-MASTER-DIRECTION-NBFC-19-10-2023.pdf \\
      --module credit --issuer RBI --doc-type master_direction --year 2023

  # Batch from directory
  python scripts/ingest_regulatory_docs.py \\
      --dir /path/to/sebi_circulars --module investment --issuer SEBI

  # Dry run
  python scripts/ingest_regulatory_docs.py \\
      --dir /path/to/docs --dry-run

  # Save report
  python scripts/ingest_regulatory_docs.py \\
      --dir /path/to/docs --module taxation --issuer CBDT \\
      --output-report ingestion_report.json

Exit codes:
  0 — all files ingested successfully (or dry-run completed)
  1 — one or more files failed
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8081")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

VALID_MODULES = {"credit", "investment", "insurance", "retirement", "taxation"}

VALID_DOC_TYPES = {
    "circular",
    "master_direction",
    "notification",
    "guideline",
    "regulation",
    "act",
    "guide",
    "factsheet",
    "policy",
    "report",
}

# Metadata templates keyed by issuer name (case-insensitive lookup)
ISSUER_TEMPLATES: Dict[str, Dict] = {
    "RBI": {
        "issuer": "RBI",
        "regulator_tag": "RBI",
        "jurisdiction": "India",
        "compliance_tags": "banking,nbfc,prudential",
        "security": "Public",
        "language": "EN",
    },
    "SEBI": {
        "issuer": "SEBI",
        "regulator_tag": "SEBI",
        "jurisdiction": "India",
        "compliance_tags": "securities,markets,mutual_funds",
        "security": "Public",
        "language": "EN",
    },
    "IRDAI": {
        "issuer": "IRDAI",
        "regulator_tag": "IRDAI",
        "jurisdiction": "India",
        "compliance_tags": "insurance,irdai",
        "security": "Public",
        "language": "EN",
    },
    "CBDT": {
        "issuer": "CBDT",
        "regulator_tag": "CBDT",
        "jurisdiction": "India",
        "compliance_tags": "taxation,income_tax",
        "security": "Public",
        "language": "EN",
    },
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest regulatory documents into the FinOrbit RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", metavar="PATH", help="Path to a single document file")
    source.add_argument("--dir", metavar="PATH", help="Directory to batch-ingest (non-recursive)")

    parser.add_argument(
        "--module",
        required=True,
        choices=sorted(VALID_MODULES),
        help="Target financial module",
    )
    parser.add_argument(
        "--issuer",
        default=None,
        choices=sorted(ISSUER_TEMPLATES.keys()) + [k.lower() for k in ISSUER_TEMPLATES],
        help="Regulatory issuer (auto-populates metadata template)",
    )
    parser.add_argument(
        "--doc-type",
        default="guideline",
        choices=sorted(VALID_DOC_TYPES),
        help="Document type (default: guideline)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now().year,
        help=f"Publication year (default: {datetime.now().year})",
    )
    parser.add_argument(
        "--jurisdiction",
        default=None,
        help="Override jurisdiction (default from issuer template or 'India')",
    )
    parser.add_argument(
        "--rag-url",
        default=RAG_API_URL,
        help=f"RAG API base URL (default: {RAG_API_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print plan without calling the API",
    )
    parser.add_argument(
        "--output-report",
        metavar="PATH",
        help="Write a JSON ingestion report to this file",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_files(args: argparse.Namespace) -> List[Path]:
    """Return a list of file paths to process."""
    if args.file:
        return [Path(args.file)]

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"[ERROR] Not a directory: {directory}")
        sys.exit(1)

    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort()
    return files


def validate_files(files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """Split into (valid, missing) lists."""
    valid, missing = [], []
    for f in files:
        (valid if f.exists() else missing).append(f)
    return valid, missing


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def build_metadata(args: argparse.Namespace) -> Dict[str, str]:
    """Build the form-data metadata dict for this ingestion run."""
    meta: Dict[str, str] = {
        "module": args.module,
        "doc_type": args.doc_type,
        "year": str(args.year),
    }

    # Apply issuer template if provided
    if args.issuer:
        template = ISSUER_TEMPLATES.get(args.issuer.upper(), {})
        meta.update(template)

    # CLI overrides
    if args.jurisdiction:
        meta["jurisdiction"] = args.jurisdiction

    # Defaults for any missing fields
    meta.setdefault("issuer", "Unknown")
    meta.setdefault("jurisdiction", "India")

    return meta


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def ingest_file(
    client: httpx.AsyncClient,
    file_path: Path,
    metadata: Dict[str, str],
    rag_url: str,
    dry_run: bool,
    max_retries: int = 3,
) -> Dict:
    """
    Ingest a single file. Returns a result dict with keys:
        file, status ("success" | "failed" | "dry_run"), chunks_created, error, duration_s
    """
    result = {
        "file": str(file_path),
        "status": "failed",
        "chunks_created": 0,
        "error": None,
        "duration_s": 0.0,
    }

    if dry_run:
        print(f"  [DRY RUN] Would ingest: {file_path.name} → module={metadata['module']}, "
              f"issuer={metadata.get('issuer')}, doc_type={metadata['doc_type']}, year={metadata['year']}")
        result["status"] = "dry_run"
        return result

    t0 = time.monotonic()

    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, _mime_type(file_path))}
                response = await client.post(
                    f"{rag_url}/ingest",
                    files=files,
                    data=metadata,
                    timeout=300.0,
                )

            if response.status_code == 200:
                body = response.json()
                result["status"] = "success"
                result["chunks_created"] = body.get("chunks_created", 0)
                result["duration_s"] = round(time.monotonic() - t0, 2)
                print(f"  [OK] {file_path.name}: {result['chunks_created']} chunks "
                      f"(job={body.get('job_id', 'N/A')}, {result['duration_s']}s)")
                return result

            # 4xx → no point retrying
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"  [ERROR] {file_path.name}: {error_msg}")
            result["error"] = error_msg
            if response.status_code < 500:
                break

        except Exception as exc:
            error_msg = str(exc)
            print(f"  [ERROR] Attempt {attempt}/{max_retries} for {file_path.name}: {error_msg}")
            result["error"] = error_msg

        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"  [RETRY] Waiting {wait}s…")
            await asyncio.sleep(wait)

    result["duration_s"] = round(time.monotonic() - t0, 2)
    return result


def _mime_type(path: Path) -> str:
    mapping = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }
    return mapping.get(path.suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(results: List[Dict], total_duration_s: float, output_path: Optional[str]) -> None:
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry_run = sum(1 for r in results if r["status"] == "dry_run")

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total": len(results),
            "success": success,
            "failed": failed,
            "dry_run": dry_run,
            "total_duration_s": round(total_duration_s, 2),
        },
        "results": results,
    }

    print("\n" + "=" * 70)
    print(f"REPORT: {success} succeeded, {failed} failed, {dry_run} dry-run "
          f"({len(results)} total, {report['summary']['total_duration_s']}s)")
    print("=" * 70)

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"Report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    args = parse_args()
    rag_url = args.rag_url

    # Collect files
    files = collect_files(args)
    if not files:
        print("[ERROR] No supported files found.")
        return 1

    valid, missing = validate_files(files)
    if missing:
        print("[ERROR] The following files do not exist:")
        for f in missing:
            print(f"  - {f}")
        return 1

    metadata = build_metadata(args)

    print("=" * 70)
    print(f"FinOrbit Regulatory Document Ingestion")
    print("=" * 70)
    print(f"  Files     : {len(valid)}")
    print(f"  Module    : {metadata['module']}")
    print(f"  Issuer    : {metadata.get('issuer', 'Unknown')}")
    print(f"  Doc type  : {metadata['doc_type']}")
    print(f"  Year      : {metadata['year']}")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  RAG URL   : {rag_url}")
    print("=" * 70)

    # Health check (skip in dry run)
    if not args.dry_run:
        try:
            async with httpx.AsyncClient(timeout=5.0) as hc:
                resp = await hc.get(f"{rag_url}/health")
            if resp.status_code != 200:
                print(f"[ERROR] RAG server unhealthy: HTTP {resp.status_code}")
                return 1
            print("[OK] RAG server is healthy\n")
        except Exception as e:
            print(f"[ERROR] Cannot reach RAG server at {rag_url}: {e}")
            print("Start it with: cd Finorbit_RAG && python main.py")
            return 1

    # Ingest
    t_start = time.monotonic()
    results: List[Dict] = []

    async with httpx.AsyncClient() as client:
        for file_path in valid:
            print(f"\n→ {file_path.name}")
            result = await ingest_file(client, file_path, metadata, rag_url, args.dry_run)
            results.append(result)
            if not args.dry_run:
                await asyncio.sleep(1)  # Brief pause between files

    total_duration = time.monotonic() - t_start
    generate_report(results, total_duration, args.output_report)

    failed_count = sum(1 for r in results if r["status"] == "failed")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
