"""Shared grading service: HTTP front end plus a single global evaluation worker.

One service instance serves every concurrent run. The single worker thread is
the whole point: evaluations are serialised process-wide, so N runs submitting
at once can never stack N model servers onto the evaluation card. Per-run
grading servers could not offer that, because they cannot see each other.

Endpoints
    POST /register    {admin_token, run_id, task, base_model, sandbox_root,
                       eval_dir, role?, log_dir?} -> {token}
    POST /unregister  {token}
    POST /submit      {token, model_path, mode?, limit?, write_metrics?}
                      -> {job_id, queue_position}
    GET  /result      ?token=...&job_id=...  -> job state and, once finished, the score
    GET  /baseline    ?token=...             -> cached baseline for (task, base_model)
    POST /baseline    {admin_token, token, score}
    GET  /queue       ?token=...             -> own jobs in full, others only counted
    GET  /health                             -> service status; host paths only with
                                                the admin token

Issuing a token is an operator action, so /register carries the admin secret from
<state_dir>/admin.token rather than being open. The agent container shares the
host network namespace and can reach this port, so without that secret it could
mint itself a token declaring role=official -- verified to work before the secret
existed -- and with it bypass the validation split, the disclosure filter and the
rule that only the official evaluation writes metrics.json.

Tokens are issued at registration and never echoed back by any read endpoint.
A submission's task and permitted directory come from its token, so one run
cannot evaluate, or even name, another run's artifacts. Reads are scoped the
same way: /result rejects a job that belongs to another token, and /queue never
discloses another run's job ids or model paths.
"""

from .app import Handler, main

from .jobs import (

    VALID_MODES,

    Conflict,

    Job,

    NO_PROGRESS_DURATION_LIMIT,

    _no_progress_share,

)

from .service import GradingService
