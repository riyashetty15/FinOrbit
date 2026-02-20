"""
Ingestion Job Management Module.

Provides database operations for tracking async document ingestion jobs.
"""

import uuid
import logging
import psycopg2
import psycopg2.extras
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

from config import get_database_config

logger = logging.getLogger(__name__)


# ============================================================================
# Database Connection Helper
# ============================================================================

def _get_connection():
    """Get a database connection"""
    db_config = get_database_config()
    return psycopg2.connect(**db_config.to_dict())


# ============================================================================
# Job CRUD Operations
# ============================================================================

def create_job(
    module: str,
    filename: str,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    file_size: Optional[int] = None
) -> str:
    """
    Create a new ingestion job record.
    
    Args:
        module: Module name (credit, investment, insurance, retirement, taxation)
        filename: Original uploaded filename
        file_path: Temporary file path for worker to process
        metadata: Ingestion metadata (doc_type, year, issuer, etc.)
        file_size: File size in bytes
    
    Returns:
        job_id: UUID string for tracking the job
    
    Raises:
        Exception: If database operation fails
    """
    job_id = str(uuid.uuid4())
    
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ingestion_jobs (
                job_id, module, filename, file_path, metadata, file_size, status
            ) VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            RETURNING id
        """, (
            job_id,
            module,
            filename,
            file_path,
            psycopg2.extras.Json(metadata) if metadata else None,
            file_size
        ))
        
        db_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Created ingestion job: job_id={job_id}, id={db_id}, module={module}, filename={filename}")
        return job_id
        
    except Exception as e:
        logger.error(f"Failed to create ingestion job: {e}", exc_info=True)
        raise


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job details by job_id.
    
    Args:
        job_id: UUID string of the job
    
    Returns:
        Dictionary with job details, or None if not found
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                id, job_id, module, filename, status,
                metadata, file_path, file_size,
                result, error_message, error_traceback,
                progress_percent, progress_message,
                document_id, worker_id, retry_count,
                created_at, started_at, completed_at, updated_at
            FROM ingestion_jobs
            WHERE job_id = %s
        """, (job_id,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            # Convert Row to dict and handle datetime serialization
            job = dict(row)
            # Convert datetime objects to ISO strings for JSON serialization
            for key in ['created_at', 'started_at', 'completed_at', 'updated_at']:
                if job.get(key):
                    job[key] = job[key].isoformat()
            return job
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}", exc_info=True)
        raise


def update_job_status(
    job_id: str,
    status: str,
    worker_id: Optional[str] = None,
    error_message: Optional[str] = None,
    error_traceback: Optional[str] = None
) -> bool:
    """
    Update job status.
    
    Args:
        job_id: UUID string of the job
        status: New status (pending, processing, completed, failed, cancelled)
        worker_id: Optional worker identifier
        error_message: Optional error message (for failed status)
        error_traceback: Optional error traceback (for failed status)
    
    Returns:
        True if update succeeded, False if job not found
    
    Raises:
        Exception: If database operation fails
    """
    valid_statuses = ['pending', 'processing', 'completed', 'failed', 'cancelled']
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
    
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        # Build dynamic UPDATE based on status
        updates = ["status = %s"]
        params = [status]
        
        if status == 'processing':
            updates.append("started_at = NOW()")
        elif status in ['completed', 'failed', 'cancelled']:
            updates.append("completed_at = NOW()")
        
        if worker_id:
            updates.append("worker_id = %s")
            params.append(worker_id)
        
        if error_message:
            updates.append("error_message = %s")
            params.append(error_message)
        
        if error_traceback:
            updates.append("error_traceback = %s")
            params.append(error_traceback)
        
        params.append(job_id)
        
        query = f"""
            UPDATE ingestion_jobs
            SET {', '.join(updates)}
            WHERE job_id = %s
        """
        
        cursor.execute(query, params)
        rows_updated = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        if rows_updated > 0:
            logger.info(f"Updated job {job_id} status to '{status}'")
            return True
        else:
            logger.warning(f"Job {job_id} not found for status update")
            return False
        
    except Exception as e:
        logger.error(f"Failed to update job status for {job_id}: {e}", exc_info=True)
        raise


def update_job_progress(
    job_id: str,
    progress_percent: int,
    progress_message: Optional[str] = None
) -> bool:
    """
    Update job progress.
    
    Args:
        job_id: UUID string of the job
        progress_percent: Progress percentage (0-100)
        progress_message: Optional progress message (e.g., "Parsing page 5/10")
    
    Returns:
        True if update succeeded, False if job not found
    
    Raises:
        ValueError: If progress_percent is out of range
        Exception: If database operation fails
    """
    if not 0 <= progress_percent <= 100:
        raise ValueError(f"progress_percent must be 0-100, got {progress_percent}")
    
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE ingestion_jobs
            SET progress_percent = %s, progress_message = %s
            WHERE job_id = %s
        """, (progress_percent, progress_message, job_id))
        
        rows_updated = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        if rows_updated > 0:
            logger.debug(f"Updated job {job_id} progress to {progress_percent}%: {progress_message}")
            return True
        else:
            logger.warning(f"Job {job_id} not found for progress update")
            return False
        
    except Exception as e:
        logger.error(f"Failed to update job progress for {job_id}: {e}", exc_info=True)
        raise


def set_job_result(
    job_id: str,
    result: Dict[str, Any],
    document_id: Optional[int] = None
) -> bool:
    """
    Set job result (when completed successfully).
    
    Args:
        job_id: UUID string of the job
        result: Ingestion result dictionary
        document_id: Optional document ID that was created
    
    Returns:
        True if update succeeded, False if job not found
    
    Raises:
        Exception: If database operation fails
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE ingestion_jobs
            SET 
                result = %s,
                document_id = %s,
                status = 'completed',
                completed_at = NOW(),
                progress_percent = 100
            WHERE job_id = %s
        """, (
            psycopg2.extras.Json(result),
            document_id,
            job_id
        ))
        
        rows_updated = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        if rows_updated > 0:
            logger.info(f"Set result for job {job_id} (document_id={document_id})")
            return True
        else:
            logger.warning(f"Job {job_id} not found for result update")
            return False
        
    except Exception as e:
        logger.error(f"Failed to set job result for {job_id}: {e}", exc_info=True)
        raise


def set_job_error(
    job_id: str,
    error_message: str,
    error_traceback: Optional[str] = None
) -> bool:
    """
    Set job error (when failed).
    
    Args:
        job_id: UUID string of the job
        error_message: Error message
        error_traceback: Optional full traceback
    
    Returns:
        True if update succeeded, False if job not found
    
    Raises:
        Exception: If database operation fails
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE ingestion_jobs
            SET 
                error_message = %s,
                error_traceback = %s,
                status = 'failed',
                completed_at = NOW()
            WHERE job_id = %s
        """, (error_message, error_traceback, job_id))
        
        rows_updated = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        if rows_updated > 0:
            logger.error(f"Set error for job {job_id}: {error_message}")
            return True
        else:
            logger.warning(f"Job {job_id} not found for error update")
            return False
        
    except Exception as e:
        logger.error(f"Failed to set job error for {job_id}: {e}", exc_info=True)
        raise


def list_jobs(
    module: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "created_at",
    order_dir: str = "DESC"
) -> List[Dict[str, Any]]:
    """
    List jobs with optional filters.
    
    Args:
        module: Filter by module name
        status: Filter by status
        limit: Maximum number of jobs to return
        offset: Offset for pagination
        order_by: Column to order by (created_at, updated_at, completed_at, status)
        order_dir: Order direction (ASC or DESC)
    
    Returns:
        List of job dictionaries
    
    Raises:
        Exception: If database operation fails
    """
    valid_order_by = ['created_at', 'updated_at', 'completed_at', 'status', 'module']
    if order_by not in valid_order_by:
        raise ValueError(f"Invalid order_by: {order_by}. Must be one of {valid_order_by}")
    
    if order_dir not in ['ASC', 'DESC']:
        raise ValueError(f"Invalid order_dir: {order_dir}. Must be ASC or DESC")
    
    try:
        conn = _get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build WHERE clause
        where_clauses = []
        params = []
        
        if module:
            where_clauses.append("module = %s")
            params.append(module)
        
        if status:
            where_clauses.append("status = %s")
            params.append(status)
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Add limit and offset
        params.extend([limit, offset])
        
        query = f"""
            SELECT 
                id, job_id, module, filename, status,
                metadata, file_size,
                result, error_message,
                progress_percent, progress_message,
                document_id, worker_id, retry_count,
                created_at, started_at, completed_at, updated_at
            FROM ingestion_jobs
            {where_sql}
            ORDER BY {order_by} {order_dir}
            LIMIT %s OFFSET %s
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert to list of dicts with datetime serialization
        jobs = []
        for row in rows:
            job = dict(row)
            # Convert datetime objects to ISO strings
            for key in ['created_at', 'started_at', 'completed_at', 'updated_at']:
                if job.get(key):
                    job[key] = job[key].isoformat()
            jobs.append(job)
        
        logger.info(f"Listed {len(jobs)} jobs (module={module}, status={status}, limit={limit})")
        return jobs
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise


def delete_old_jobs(
    days_old: int = 30,
    statuses: Optional[List[str]] = None
) -> int:
    """
    Delete old completed/failed/cancelled jobs.
    
    Args:
        days_old: Delete jobs older than this many days
        statuses: Optional list of statuses to delete (default: completed, failed, cancelled)
    
    Returns:
        Number of jobs deleted
    
    Raises:
        Exception: If database operation fails
    """
    if statuses is None:
        statuses = ['completed', 'failed', 'cancelled']
    
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cursor.execute("""
            DELETE FROM ingestion_jobs
            WHERE created_at < %s
              AND status = ANY(%s)
        """, (cutoff_date, statuses))
        
        rows_deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Deleted {rows_deleted} old jobs (older than {days_old} days, statuses={statuses})")
        return rows_deleted
        
    except Exception as e:
        logger.error(f"Failed to delete old jobs: {e}", exc_info=True)
        raise


# ============================================================================
# Convenience Functions
# ============================================================================

def get_job_status_summary(module: Optional[str] = None) -> Dict[str, int]:
    """
    Get count of jobs by status.
    
    Args:
        module: Optional module filter
    
    Returns:
        Dictionary with status counts {status: count}
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        if module:
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM ingestion_jobs
                WHERE module = %s
                GROUP BY status
            """, (module,))
        else:
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM ingestion_jobs
                GROUP BY status
            """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        summary = {status: count for status, count in rows}
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get job status summary: {e}", exc_info=True)
        raise

