"""
FinOrbit Drift Detection — reads daily audit logs and alerts on quality degradation.

Run as a nightly cron job:
    0 6 * * * python /opt/finorbit/Finorbit_LLM/scripts/drift_check.py >> /var/log/finorbit_drift.log 2>&1

Or as a one-off check:
    python scripts/drift_check.py --date 2024-01-15 --log-dir logs/audit
"""

import argparse
import json
import os
import smtplib
import statistics
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText


# ── Alert thresholds ──────────────────────────────────────────────────────────
THRESHOLDS = {
    "min_avg_confidence":     0.60,   # alert if avg confidence drops below 60%
    "max_block_rate":         0.15,   # alert if >15% of queries are blocked
    "max_error_rate":         0.10,   # alert if >10% of events are errors
    "min_query_count":        1,      # alert if no queries logged (service down?)
    "max_circuit_breaker_rate": 0.05, # alert if >5% of LLM calls trip circuit breaker
}


def load_audit_log(log_dir: str, date: str) -> list:
    """Load all entries from a single day's audit log."""
    log_file = os.path.join(log_dir, f"audit_{date}.jsonl")

    if not os.path.exists(log_file):
        print(f"[drift] No audit log found for {date} at {log_file}")
        return []

    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[drift] WARNING: malformed JSON on line {line_num}, skipping")

    print(f"[drift] Loaded {len(entries)} audit entries for {date}")
    return entries


def analyse(entries: list, date: str) -> dict:
    """Compute drift metrics from audit entries."""
    if not entries:
        return {"date": date, "total_events": 0, "error": "no_data"}

    total = len(entries)
    query_events = [e for e in entries if e.get("event_type") == "query_received"]
    validation_events = [e for e in entries if e.get("event_type") == "validation_completed"]
    blocked_events = [e for e in entries if e.get("action_taken") in ("blocked", "refused")]
    error_events = [e for e in entries if e.get("severity") == "critical"]

    # Confidence scores
    confidence_scores = [
        e["details"].get("confidence_score")
        for e in validation_events
        if isinstance(e.get("details", {}).get("confidence_score"), (int, float))
    ]

    avg_confidence = statistics.mean(confidence_scores) if confidence_scores else None
    block_rate = len(blocked_events) / len(query_events) if query_events else 0
    error_rate = len(error_events) / total if total else 0

    # Top routed domains
    domain_counts: dict = {}
    for e in entries:
        module = e.get("details", {}).get("module") or e.get("details", {}).get("agent")
        if module:
            domain_counts[module] = domain_counts.get(module, 0) + 1
    top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "date": date,
        "total_events": total,
        "query_count": len(query_events),
        "validation_count": len(validation_events),
        "blocked_count": len(blocked_events),
        "error_count": len(error_events),
        "avg_confidence": avg_confidence,
        "block_rate": block_rate,
        "error_rate": error_rate,
        "top_domains": top_domains,
    }


def check_thresholds(metrics: dict) -> list:
    """Return list of alert messages for any breached thresholds."""
    alerts = []

    if metrics.get("error") == "no_data":
        alerts.append("CRITICAL: No audit log data found — service may be down or logging is broken.")
        return alerts

    query_count = metrics.get("query_count", 0)
    if query_count < THRESHOLDS["min_query_count"]:
        alerts.append(f"CRITICAL: Only {query_count} queries logged — service may be down.")

    avg_conf = metrics.get("avg_confidence")
    if avg_conf is not None and avg_conf < THRESHOLDS["min_avg_confidence"]:
        alerts.append(
            f"WARNING: Average confidence {avg_conf:.2%} is below threshold "
            f"{THRESHOLDS['min_avg_confidence']:.0%} — model quality may be degrading."
        )

    block_rate = metrics.get("block_rate", 0)
    if block_rate > THRESHOLDS["max_block_rate"]:
        alerts.append(
            f"WARNING: Block rate {block_rate:.2%} exceeds threshold "
            f"{THRESHOLDS['max_block_rate']:.0%} — guardrails may be over-triggering."
        )

    error_rate = metrics.get("error_rate", 0)
    if error_rate > THRESHOLDS["max_error_rate"]:
        alerts.append(
            f"WARNING: Error rate {error_rate:.2%} exceeds threshold "
            f"{THRESHOLDS['max_error_rate']:.0%} — check LLM provider and RAG service."
        )

    return alerts


def print_report(metrics: dict, alerts: list):
    """Print a human-readable drift report."""
    print("\n" + "=" * 60)
    print(f"FINORBIT DRIFT REPORT — {metrics['date']}")
    print("=" * 60)
    print(f"  Total events:        {metrics.get('total_events', 0)}")
    print(f"  Queries:             {metrics.get('query_count', 0)}")
    print(f"  Blocked:             {metrics.get('blocked_count', 0)}  "
          f"({metrics.get('block_rate', 0):.1%})")
    print(f"  Errors (critical):   {metrics.get('error_count', 0)}  "
          f"({metrics.get('error_rate', 0):.1%})")

    avg_conf = metrics.get("avg_confidence")
    if avg_conf is not None:
        print(f"  Avg confidence:      {avg_conf:.2%}")
    else:
        print("  Avg confidence:      N/A")

    top = metrics.get("top_domains", [])
    if top:
        print("\n  Top routed domains:")
        for domain, count in top:
            print(f"    {domain:<25} {count}")

    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    ⚠ {a}")
    else:
        print("\n  All metrics within normal range.")

    print("=" * 60)


def send_email_alert(alerts: list, metrics: dict):
    """
    Send email alert via Outlook SMTP.
    Requires env vars:
        ALERT_EMAIL_FROM     — your Outlook address (e.g. you@outlook.com)
        ALERT_EMAIL_PASSWORD — your Outlook password (or App Password if MFA is on)
        ALERT_EMAIL_TO       — recipient address
    """
    sender    = os.getenv("ALERT_EMAIL_FROM")
    password  = os.getenv("ALERT_EMAIL_PASSWORD")
    recipient = os.getenv("ALERT_EMAIL_TO")

    if not all([sender, password, recipient]):
        print("[drift] Email alert skipped — ALERT_EMAIL_* env vars not set.")
        return

    date    = metrics.get("date", "unknown")
    subject = f"[FinOrbit] Drift Alert — {len(alerts)} issue(s) detected on {date}"

    body_lines = [
        f"FinOrbit drift check for {date} found {len(alerts)} alert(s):\n",
        *[f"  • {a}" for a in alerts],
        "",
        f"  Queries processed:  {metrics.get('query_count', 0)}",
        f"  Block rate:         {metrics.get('block_rate', 0):.1%}",
        f"  Error rate:         {metrics.get('error_rate', 0):.1%}",
    ]
    avg_conf = metrics.get("avg_confidence")
    if avg_conf is not None:
        body_lines.append(f"  Avg confidence:     {avg_conf:.2%}")

    msg            = MIMEText("\n".join(body_lines))
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient

    try:
        # Outlook uses port 587 with STARTTLS (not SSL like Gmail)
        with smtplib.SMTP("smtp.office365.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(sender, password)
            smtp.sendmail(sender, recipient, msg.as_string())
        print(f"[drift] Email alert sent to {recipient}")
    except Exception as e:
        print(f"[drift] Failed to send email: {e}")


def send_slack_alert(alerts: list, metrics: dict):
    """
    Send Slack alert via incoming webhook.
    Requires env var:
        SLACK_WEBHOOK_URL — from Slack App → Incoming Webhooks
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("[drift] Slack alert skipped — SLACK_WEBHOOK_URL not set.")
        return

    date = metrics.get("date", "unknown")
    alert_lines = "\n".join(f"• {a}" for a in alerts)
    avg_conf = metrics.get("avg_confidence")
    conf_str = f"{avg_conf:.2%}" if avg_conf is not None else "N/A"

    payload = {
        "text": (
            f":warning: *FinOrbit Drift Alert — {date}*\n\n"
            f"*{len(alerts)} issue(s) detected:*\n{alert_lines}\n\n"
            f"Queries: {metrics.get('query_count', 0)} | "
            f"Block rate: {metrics.get('block_rate', 0):.1%} | "
            f"Avg confidence: {conf_str}"
        )
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        print("[drift] Slack alert sent.")
    except Exception as e:
        print(f"[drift] Failed to send Slack alert: {e}")


def main():
    parser = argparse.ArgumentParser(description="FinOrbit nightly drift check")
    parser.add_argument(
        "--date",
        default=None,
        help="Date to check in YYYY-MM-DD format (default: yesterday)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/audit",
        help="Path to audit log directory (default: logs/audit)",
    )
    args = parser.parse_args()

    if args.date:
        check_date = args.date
    else:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        check_date = yesterday.strftime("%Y-%m-%d")

    entries = load_audit_log(args.log_dir, check_date)
    metrics = analyse(entries, check_date)
    alerts = check_thresholds(metrics)
    print_report(metrics, alerts)

    # Send notifications if any alerts found
    if alerts:
        send_email_alert(alerts, metrics)
        send_slack_alert(alerts, metrics)

    # Exit 1 if any CRITICAL alerts — cron job can use this to send notifications
    if any("CRITICAL" in a for a in alerts):
        sys.exit(1)


if __name__ == "__main__":
    main()
