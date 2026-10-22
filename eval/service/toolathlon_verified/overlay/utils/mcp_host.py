"""Central resolver for the MCP host address.

Toolathlon-Verified task configs and preprocess/eval scripts address the
app-side services (poste IMAP/SMTP, woocommerce, canvas, ...) with
``localhost`` -- the upstream single-machine assumption. When the agent runs
inside a bridge-network container, ``localhost`` points at the container
itself, not the host-bound app services.

Instead of hardcoding the host IP into every task config, set
``TOOLATHLON_MCP_HOST`` (environment variable; runner.py injects the docker0
gateway 172.17.0.1 by default) or ``mcp_host`` (in ``configs/global_configs.py``)
on the task machine. The framework then rewrites ``localhost`` to that host at
the consumption points (token sessions, email configs, task text, ...).
Unset -> no rewriting, i.e. upstream single-machine (localhost) semantics.
"""
import json
import os
import re
from typing import Any, Dict, Optional

# Environment variable naming the MCP host for reaching host-bound services
# from inside a bridge-network container.
ENV_VAR = "TOOLATHLON_MCP_HOST"

# A "localhost" token: not part of a larger word, not preceded by '.' (a
# subdomain) or '@' (an email address). A trailing ':port' is allowed, so both
# bare hosts ("imap_server": "localhost") and URLs (http://localhost:10003)
# are rewritten, while strings like "mylocalhost" / "localhost-1" /
# "user@localhost" / "foo.localhost" stay untouched.
_LOCALHOST_RE = re.compile(r"(?<![\w.@-])localhost(?![\w-])")

# Email-account config fields that name a server host.
_EMAIL_SERVER_FIELDS = ("imap_server", "smtp_server")


def get_mcp_host() -> Optional[str]:
    """Return the host to substitute for ``localhost``, or None (keep as-is).

    Priority: $TOOLATHLON_MCP_HOST, then ``mcp_host`` in
    configs/global_configs.py. The global_configs import is lazy/optional so
    this module also works in environments without a filled-in configs dir.
    """
    host = (os.environ.get(ENV_VAR) or "").strip()
    if host:
        return host
    try:
        from configs.global_configs import global_configs
        host = str(global_configs.get("mcp_host") or "").strip()
        if host:
            return host
    except Exception:
        # configs/global_configs.py may be absent (e.g. bare test envs).
        pass
    return None


def remap_localhost(value: Any, mcp_host: Optional[str] = None) -> Any:
    """Recursively rewrite ``localhost`` host tokens to the MCP host.

    Walks str / list / dict (dict subclasses such as addict.Dict are
    reconstructed with their own type). No-op when no MCP host is configured.
    """
    if mcp_host is None:
        mcp_host = get_mcp_host()
    if not mcp_host:
        return value

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return _LOCALHOST_RE.sub(mcp_host, obj)
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        if isinstance(obj, dict):
            return type(obj)((key, _walk(item)) for key, item in obj.items())
        return obj

    return _walk(value)


def remap_email_config_fields(config: Dict[str, Any], mcp_host: Optional[str] = None) -> Dict[str, Any]:
    """Remap the server-host fields of an email account config, in place."""
    if mcp_host is None:
        mcp_host = get_mcp_host()
    if not mcp_host:
        return config
    for key in _EMAIL_SERVER_FIELDS:
        if config.get(key) == "localhost":
            config[key] = mcp_host
    return config


def resolve_email_config_for_host(config_file: str, out_dir: str,
                                  mcp_host: Optional[str] = None) -> Optional[str]:
    """Copy an emails JSON config with localhost servers remapped to the MCP host.

    Returns the copy's path, or None when no rewrite is needed (no MCP host
    configured, file unreadable, or nothing localhost in it). Used for the
    local emails-mcp server in split mode: emails-mcp reads the JSON itself,
    so the rewrite cannot happen at the LocalEmailManager layer. Handles both
    a single account object and nested account collections.
    """
    if mcp_host is None:
        mcp_host = get_mcp_host()
    if not mcp_host:
        return None
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None

    def _remap(node: Any) -> bool:
        changed = False
        if isinstance(node, dict):
            for key in _EMAIL_SERVER_FIELDS:
                if node.get(key) == "localhost":
                    node[key] = mcp_host
                    changed = True
            for item in node.values():
                changed = _remap(item) or changed
        elif isinstance(node, list):
            for item in node:
                changed = _remap(item) or changed
        return changed

    if not _remap(data):
        return None
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "local_emails_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path
