# ==============================================
# File: backend/agents/safety/audit_logger.py
# Description: Audit logging agent for compliance
# ==============================================

from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import json
import os
from backend.agents.safety.base_safety import BaseSafetyAgent
from backend.core.validation_models import AuditEntry, Severity


class AuditLoggerAgent(BaseSafetyAgent):
    """
    Structured audit logging for compliance (Module 2, Safety Agent #5)

    Logs all validation and safety events in structured JSON format:
    - Query text (sanitized)
    - Retrieved sources
    - Draft answer
    - Validation outcomes
    - Safety flags
    - Confidence scores
    - Actions taken (blocked, sanitized, served)

    **Action**: Write audit trail entries (non-blocking)
    **Severity**: INFO - For compliance and monitoring
    """

    def __init__(self, log_dir: str = "logs/audit"):
        """
        Initialize audit logger

        Args:
            log_dir: Directory to store audit log files
        """
        super().__init__(name="audit_logger")
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Current log file path (daily rotation)
        self.log_file = self._get_log_file_path()

    def _get_log_file_path(self) -> str:
        """Get log file path with daily rotation"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit_{today}.jsonl")

    def check(self, query: str, profile: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Log query for audit trail (pre-execution)

        Args:
            query: User query text
            profile: User profile

        Returns:
            Tuple of (is_safe, issues, metadata):
                - is_safe: Always True (logging doesn't block)
                - issues: Empty list
                - metadata: {"logged": True, "log_file": path}
        """
        # Create audit entry
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="query_received",
            agent_name=self.name,
            details={
                "query": query,
                "query_length": len(query),
                "user_id": profile.get("user_id", "unknown"),
                "profile_summary": {
                    "age": profile.get("age"),
                    "income": profile.get("income"),
                }
            },
            severity=Severity.INFO,
            action_taken="logged"
        )

        # Write to log file
        self._write_audit_entry(entry)

        # Always safe (non-blocking)
        metadata = {
            "logged": True,
            "log_file": self.log_file
        }

        return self._create_result(True, [], metadata)

    def log_validation_results(
        self,
        query: str,
        response: str,
        validation_results: Dict[str, Any],
        confidence_score: float,
        action_taken: str
    ) -> None:
        """
        Log validation results (post-execution)

        Args:
            query: User query
            response: Agent response
            validation_results: Validation check results
            confidence_score: Overall confidence score
            action_taken: Action taken (served, blocked, etc.)
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="validation_completed",
            agent_name=self.name,
            details={
                "query": query[:200],  # Truncate for log size
                "response_length": len(response),
                "validation_summary": validation_results,
                "confidence_score": confidence_score,
            },
            severity=self._determine_severity(action_taken),
            action_taken=action_taken
        )

        self._write_audit_entry(entry)

    def log_event(
        self,
        event_type: str,
        query: str,
        profile: Dict[str, Any],
        details: Dict[str, Any]
    ) -> AuditEntry:
        """
        Generic event logging method

        Args:
            event_type: Type of event
            query: Query text
            profile: User profile
            details: Event details

        Returns:
            AuditEntry object
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            agent_name=self.name,
            details=details,
            severity=Severity.INFO,
            action_taken="logged"
        )

        self._write_audit_entry(entry)
        return entry

    def log_safety_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: Severity,
        action_taken: str
    ) -> None:
        """
        Log safety event (PII detection, content risk, etc.)

        Args:
            event_type: Type of safety event
            details: Event details
            severity: Severity level
            action_taken: Action taken
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            agent_name=self.name,
            details=details,
            severity=severity,
            action_taken=action_taken
        )

        self._write_audit_entry(entry)

    def _write_audit_entry(self, entry: AuditEntry) -> None:
        """
        Write audit entry to JSONL log file

        Args:
            entry: AuditEntry to write
        """
        try:
            # Rotate log file if date changed
            current_log_file = self._get_log_file_path()
            if current_log_file != self.log_file:
                self.log_file = current_log_file

            # Convert entry to dict for JSON serialization
            entry_dict = {
                "timestamp": entry.timestamp,
                "event_type": entry.event_type,
                "agent_name": entry.agent_name,
                "details": entry.details,
                "severity": entry.severity.value if isinstance(entry.severity, Severity) else entry.severity,
                "action_taken": entry.action_taken
            }

            # Write as JSONL (one JSON object per line)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry_dict) + "\n")

        except Exception as e:
            # If audit logging fails, don't crash the system
            print(f"[AuditLogger] Failed to write audit entry: {e}")

    def _determine_severity(self, action_taken: str) -> Severity:
        """Determine severity based on action taken"""
        if action_taken in ["blocked", "refused"]:
            return Severity.CRITICAL
        elif action_taken in ["sanitized", "warned"]:
            return Severity.WARNING
        else:
            return Severity.INFO

    def get_audit_summary(self, date: str = None) -> Dict[str, Any]:
        """
        Get audit summary for a specific date

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            Summary statistics
        """
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        log_file = os.path.join(self.log_dir, f"audit_{date}.jsonl")

        if not os.path.exists(log_file):
            return {"date": date, "total_events": 0}

        # Read and parse audit log
        events = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except:
                    continue

        # Calculate summary statistics
        summary = {
            "date": date,
            "total_events": len(events),
            "event_types": {},
            "severity_counts": {
                "critical": 0,
                "warning": 0,
                "info": 0
            },
            "actions_taken": {}
        }

        for event in events:
            # Count event types
            event_type = event.get("event_type", "unknown")
            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1

            # Count severity
            severity = event.get("severity", "info")
            if severity in summary["severity_counts"]:
                summary["severity_counts"][severity] += 1

            # Count actions
            action = event.get("action_taken", "unknown")
            summary["actions_taken"][action] = summary["actions_taken"].get(action, 0) + 1

        return summary
