"""Run manifest: everything needed to reproduce and audit a batch run.

Records library version, Python version, dependency versions,
platform, random seed, configuration digest, timestamp, and per-step
status — enough to reproduce and audit a run without reconstructing it
from conversation/developer memory.

Notes
-----
Never includes access tokens, home-directory secrets, raw environment
variables, or other credentials.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

__all__ = ["build_manifest", "config_digest"]


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "scipy", "statsmodels"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not installed"
    for optional_name in ("matplotlib", "pandas"):
        try:
            module = __import__(optional_name)
            versions[optional_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def config_digest(configuration: Mapping[str, Any]) -> str:
    """Compute a stable digest of a resolved configuration mapping.

    Parameters
    ----------
    configuration : Mapping[str, Any]
        The resolved configuration to fingerprint. Values that are not
        natively JSON-serializable are stringified via ``default=str``.

    Returns
    -------
    str
        A 64-character hexadecimal SHA-256 digest. Keys are sorted
        before hashing, so two configurations that differ only in key
        order hash identically.

    Examples
    --------
    >>> from mrsvalidate.io import config_digest
    >>> d1 = config_digest({"a": 1, "b": 2})
    >>> d2 = config_digest({"b": 2, "a": 1})
    >>> d1 == d2
    True
    >>> len(d1)
    64
    """

    encoded = json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *,
    step_names: list[str],
    step_statuses: Mapping[str, str],
    resolved_configuration: Mapping[str, Any],
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Build a self-describing manifest for one batch run.

    Parameters
    ----------
    step_names : list of str
        Names of every step in the run plan, in execution order.
    step_statuses : Mapping[str, str]
        Mapping from step name to its outcome status string (e.g.
        ``"valid"`` or ``"failed_exception"``).
    resolved_configuration : Mapping[str, Any]
        The fully resolved configuration for the run, used to compute
        ``config_digest`` via :func:`config_digest`.
    random_seed : int or None, optional
        Random seed used for the run, if any. Default is ``None``.

    Returns
    -------
    dict
        A JSON-serializable manifest with keys ``library_version``,
        ``python_version``, ``dependency_versions``, ``platform``,
        ``timestamp_utc``, ``random_seed``, ``config_digest``,
        ``steps``, and ``step_statuses``.

    Notes
    -----
    Never includes access tokens, home-directory secrets, raw
    environment variables, or other credentials — only version,
    platform, and configuration-provenance information.

    Examples
    --------
    >>> from mrsvalidate.io import build_manifest
    >>> manifest = build_manifest(
    ...     step_names=["regression"],
    ...     step_statuses={"regression": "valid"},
    ...     resolved_configuration={"n_predictors": 1},
    ...     random_seed=0,
    ... )
    >>> sorted(manifest.keys())
    ['config_digest', 'dependency_versions', 'library_version', 'platform', 'python_version', 'random_seed', 'step_statuses', 'steps', 'timestamp_utc']
    >>> manifest["steps"]
    ['regression']
    """

    from .. import __version__

    return {
        "library_version": __version__,
        "python_version": sys.version,
        "dependency_versions": _dependency_versions(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "config_digest": config_digest(resolved_configuration),
        "steps": step_names,
        "step_statuses": dict(step_statuses),
    }
