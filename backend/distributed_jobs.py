"""Redis/RQ transport while PostgreSQL remains the source of job truth."""
from __future__ import annotations

from .platform_contracts import QueueReceipt


def execute_ingestion_job(job_id: str) -> None:
    """Worker entry point; the runtime service resolves and leases the job by ID."""
    from .platform_runtime import get_platform_runtime

    runtime = get_platform_runtime(required=True)
    runtime.process_ingestion_job(job_id)


class RqTaskQueue:
    def __init__(self, redis_connection, queue_name: str = "ingestion"):
        from rq import Queue

        self.redis = redis_connection
        self.queue = Queue(queue_name, connection=redis_connection, default_timeout=1800)

    def enqueue(self, job_id: str) -> QueueReceipt:
        queued = self.queue.enqueue(
            execute_ingestion_job,
            job_id,
            job_id=job_id,
            retry=None,
            result_ttl=86400,
            failure_ttl=604800,
        )
        return QueueReceipt(queued.id, self.queue.name)

    def cancel(self, job_id: str) -> bool:
        from rq.command import send_stop_job_command
        from rq.job import Job

        try:
            job = Job.fetch(job_id, connection=self.redis)
        except Exception:
            return False
        if job.get_status(refresh=True) == "started":
            send_stop_job_command(self.redis, job.id)
        else:
            job.cancel()
        return True

    def health(self):
        try:
            return {"ready": bool(self.redis.ping()), "queue": self.queue.name, "queued": self.queue.count}
        except Exception:
            return {"ready": False, "queue": self.queue.name, "queued": None}
