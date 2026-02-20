"""
Background Worker for Async Document Ingestion.

Processes ingestion jobs submitted to the queue without blocking the HTTP request.
"""

import os
import logging
import traceback
from typing import Dict, Any, Optional

from ingestion.db_ingestion_models import (
    get_job,
    update_job_status,
    update_job_progress,
    set_job_result,
    set_job_error
)

logger = logging.getLogger(__name__)


# ============================================================================
# Worker Functions
# ============================================================================

def process_ingestion_job(
    job_id: str,
    pipeline,  # TextIngestionPipeline instance
    worker_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single ingestion job in the background.
    
    This function:
    1. Updates job status to 'processing'
    2. Calls pipeline.ingest() to process the document
    3. Updates job with result or error
    4. Cleans up temp file
    
    Args:
        job_id: UUID of the job to process
        pipeline: TextIngestionPipeline instance for the module
        worker_id: Optional worker identifier for tracking
    
    Returns:
        Result dictionary from pipeline.ingest()
    
    Raises:
        Exception: If job not found or processing fails critically
    """
    if worker_id is None:
        worker_id = f"worker-{os.getpid()}"
    
    logger.info(f"[{worker_id}] Starting processing for job {job_id}")
    
    # Get job details
    job = get_job(job_id)
    if not job:
        error_msg = f"Job {job_id} not found"
        logger.error(f"[{worker_id}] {error_msg}")
        raise ValueError(error_msg)
    
    file_path = job.get('file_path')
    metadata = job.get('metadata', {})
    module = job.get('module')
    
    if not file_path or not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(f"[{worker_id}] {error_msg}")
        set_job_error(job_id, error_msg)
        return {"status": "failed", "error": error_msg}
    
    try:
        # Update status to processing
        update_job_status(job_id, 'processing', worker_id=worker_id)
        update_job_progress(job_id, 10, "Starting document processing...")
        
        logger.info(f"[{worker_id}] Processing document: {file_path} (module: {module})")
        
        # Call pipeline ingestion (this is the long-running operation)
        # Note: pipeline.ingest() is synchronous, but we're running in background
        result = pipeline.ingest(
            document_path=file_path,
            metadata=metadata
        )
        
        logger.info(f"[{worker_id}] Ingestion completed for job {job_id}: {result.get('status')}")
        
        # Extract document_id from result if available
        document_id = None
        if result.get('status') == 'success' and 'document' in result:
            document_id = result['document'].get('id')
        
        # Update job with result
        if result.get('status') in ['success', 'duplicate']:
            update_job_progress(job_id, 100, "Completed successfully")
            set_job_result(job_id, result, document_id=document_id)
        else:
            # Failed status
            error_msg = result.get('message', 'Unknown error')
            set_job_error(job_id, error_msg)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(
            f"[{worker_id}] Failed to process job {job_id}: {error_msg}\n{error_trace}",
            exc_info=True
        )
        
        # Update job with error
        set_job_error(job_id, error_msg, error_traceback=error_trace)
        
        return {
            "status": "failed",
            "error": error_msg,
            "traceback": error_trace
        }
        
    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"[{worker_id}] Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"[{worker_id}] Failed to delete temp file {file_path}: {e}")


# ============================================================================
# Thread/Async Helpers
# ============================================================================

def submit_job_to_background(
    job_id: str,
    pipeline,
    executor=None,
    worker_id: Optional[str] = None
):
    """
    Submit a job to background execution.
    
    Can use ThreadPoolExecutor, ProcessPoolExecutor, or asyncio tasks.
    
    Args:
        job_id: UUID of the job to process
        pipeline: TextIngestionPipeline instance
        executor: Optional executor (ThreadPoolExecutor or ProcessPoolExecutor)
        worker_id: Optional worker identifier
    
    Returns:
        Future object if executor provided, else None
    """
    if executor:
        # Submit to thread/process pool
        future = executor.submit(
            process_ingestion_job,
            job_id=job_id,
            pipeline=pipeline,
            worker_id=worker_id
        )
        logger.info(f"Submitted job {job_id} to executor")
        return future
    else:
        # Run in separate thread (simple approach)
        import threading
        thread = threading.Thread(
            target=process_ingestion_job,
            args=(job_id, pipeline),
            kwargs={'worker_id': worker_id},
            daemon=True
        )
        thread.start()
        logger.info(f"Started background thread for job {job_id}")
        return thread


# ============================================================================
# FastAPI BackgroundTasks Integration
# ============================================================================

def create_background_task_wrapper(job_id: str, pipeline, worker_id: Optional[str] = None):
    """
    Create a wrapper function for FastAPI BackgroundTasks.
    
    Usage:
        from fastapi import BackgroundTasks
        
        @app.post("/ingest")
        async def ingest(background_tasks: BackgroundTasks, ...):
            job_id = create_job(...)
            task = create_background_task_wrapper(job_id, pipeline)
            background_tasks.add_task(task)
            return {"job_id": job_id, "status": "pending"}
    
    Args:
        job_id: UUID of the job to process
        pipeline: TextIngestionPipeline instance
        worker_id: Optional worker identifier
    
    Returns:
        Callable function for BackgroundTasks
    """
    def task():
        process_ingestion_job(job_id, pipeline, worker_id=worker_id)
    
    return task

