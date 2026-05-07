from __future__ import annotations

import logging
import os
import queue
import secrets
import threading
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
import uvicorn

from handler import check_server, handler as run_qwen_job


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("qwen-v19-pod-api")

POD_API_KEY = os.environ.get("POD_API_KEY", "").strip()
POD_API_PORT = int(os.environ.get("POD_API_PORT", "8000"))
JOB_QUEUE: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
EXECUTION_LOCK = threading.Lock()

app = FastAPI(title="Qwen v19 Clothing Edit Pod API", version="1.0.0")


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not POD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="POD_API_KEY is not configured.",
        )

    expected = f"Bearer {POD_API_KEY}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


def set_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        return dict(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def execute_job(job_id: str, job_input: dict[str, Any]) -> None:
    with EXECUTION_LOCK:
        started_at = time.time()
        set_job(
            job_id,
            id=job_id,
            status="IN_PROGRESS",
            delayTime=0,
            executionTime=0,
            output=None,
            error=None,
        )

        try:
            output = run_qwen_job({"id": job_id, "input": job_input})
            execution_ms = int((time.time() - started_at) * 1000)
            if isinstance(output, dict) and output.get("error"):
                set_job(
                    job_id,
                    status="FAILED",
                    executionTime=execution_ms,
                    error=str(output["error"]),
                    output=None,
                )
                return

            set_job(
                job_id,
                status="COMPLETED",
                executionTime=execution_ms,
                output=output,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Pod job %s failed", job_id)
            set_job(
                job_id,
                status="FAILED",
                executionTime=int((time.time() - started_at) * 1000),
                output=None,
                error=str(exc),
            )


def worker_loop() -> None:
    while True:
        job_id, job_input = JOB_QUEUE.get()
        try:
            execute_job(job_id, job_input)
        finally:
            JOB_QUEUE.task_done()


def start_worker() -> None:
    worker = threading.Thread(target=worker_loop, daemon=True, name="qwen-pod-worker")
    worker.start()


@app.on_event("startup")
def on_startup() -> None:
    start_worker()


@app.get("/health")
def health() -> dict[str, Any]:
    comfy_ready = True
    error = None
    try:
        check_server()
    except Exception as exc:  # noqa: BLE001
        comfy_ready = False
        error = str(exc)

    return {
        "status": "ready" if comfy_ready else "starting",
        "comfy_ready": comfy_ready,
        "auth_configured": bool(POD_API_KEY),
        "queued_jobs": JOB_QUEUE.qsize(),
        "error": error,
    }


@app.post("/run")
def run_async(payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    job_input = payload.get("input")
    if not isinstance(job_input, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must include an input object.")

    job_id = str(uuid.uuid4())
    set_job(
        job_id,
        id=job_id,
        status="IN_QUEUE",
        delayTime=0,
        executionTime=0,
        output=None,
        error=None,
    )
    JOB_QUEUE.put((job_id, job_input))
    return {"id": job_id, "status": "IN_QUEUE"}


@app.get("/status/{job_id}")
def status_for_job(job_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job id.")
    return job


@app.post("/runsync")
def run_sync(payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    job_input = payload.get("input")
    if not isinstance(job_input, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must include an input object.")

    job_id = str(uuid.uuid4())
    execute_job(job_id, job_input)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Job result disappeared.")
    return job


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=POD_API_PORT)
