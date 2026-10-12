"""**Validation** and **atomic writing** for the experiment pickle cache.

Background
----------
The experiment runner used to skip recomputation based on ``os.path.isfile`` alone, while
the cache key was a **directory path** that did not encode ``-t`` (the trajectory count).
Measured: the ``seed=3/period=100`` directory held only **5** trajectories, and re-running
with ``-t 100`` was silently skipped -- so a figure claiming 100 trajectories was drawn
from 5. And ``data/**`` is in ``.gitignore``, so **overwriting loses it permanently**.

This module replaces "skip if the file exists" with "skip only if four gates pass":

1. **Content validation** -- the pickle reads back, the key set matches exactly, and every
   array's first axis is ``n_traj`` (per-trajectory) or ``T+1`` (per-step parameters
   shared across trajectories). That is enough to catch truncated/half-written files and
   files with the wrong trajectory count, without a hardcoded shape table per file.
2. **Fingerprint validation** -- ``_cache_manifest.json`` in the output directory records
   ``fingerprint = cache format version + source hash + parameter snapshot``. Any change
   invalidates.
3. **Dependency validation** -- predictor results depend on ``traj.pkl`` (which carries the
   shared ``sds_params``). Saving records each dependency's sha256; loading compares them,
   so a changed dependency recomputes this file.
4. **Atomic writing** -- write ``.tmp`` then ``os.replace``, which rules out half-written
   files at the root.

**The directory layout is unchanged** (still ``data/.../seed=N/period=P``): the fingerprint
lives only in the manifest, so the existing ``data/`` tree keeps its names.

Usage
-----
::

    ctx = CacheContext(dir_out, snapshot={...}, force=args.force)
    data = ctx.load('nominal.pkl', KEYS, n_traj, T, deps=['traj.pkl'])
    if data is None:
        data = ...                      # actually recompute
        ctx.save(data, 'nominal.pkl', deps=['traj.pkl'])
    print(ctx.summary())
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

import numpy as np

MANIFEST_NAME = '_cache_manifest.json'
CACHE_FORMAT_VERSION = 1

#: Pure plotting code, which cannot affect pickle contents, and is therefore left out of
#: the source fingerprint. ⚠️ This is a **module-name prefix**, not a path. Note that
#: :func:`source_fingerprint` scans exactly ``ppsds.*``, so if plotting code were moved out
#: of ``ppsds.visualization`` this prefix would have to move with it -- otherwise editing a
#: plot would needlessly invalidate every cache (at a cost of thousands of cone programs).
_SOURCE_PREFIX_EXCLUDE = ('ppsds.visualization',)


# ----------------------------------------------------------------------
# Fingerprint
# ----------------------------------------------------------------------

def _jsonable(obj):
    """Convert to a JSON-serializable form; non-serializable values (e.g. callables) keep only their type name."""
    if isinstance(obj, np.ndarray):
        return {'__ndarray__': obj.tolist(), 'dtype': str(obj.dtype)}
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return f'<{type(obj).__name__}>'


def source_fingerprint():
    """Hash of the ``__main__`` and ``ppsds.*`` sources loaded in this process.

    It walks ``sys.modules`` rather than a hardcoded file list, so it is sensitive only to
    code that is **actually used**: editing a file nobody imported (dead code, say) does not
    needlessly invalidate the cache -- and that costs thousands of cone programs. It also
    works on a **dirty working tree**: a git hash does not change before you commit, whereas
    this does, and that is precisely when the cache must be invalidated.

    ⚠️ **When to call**: because it reads the "already loaded" set, it must be called
    **after all imports are finished**. ``CacheContext`` is constructed inside ``__main__``,
    which satisfies that naturally. Calling it midway through import would make the same
    code produce different fingerprints.
    """
    h = hashlib.sha256()
    for name in sorted(sys.modules):
        if name != '__main__' and not name.startswith('ppsds.'):
            continue
        if name.startswith(_SOURCE_PREFIX_EXCLUDE):
            continue
        path = getattr(sys.modules[name], '__file__', None)
        if not path or not os.path.isfile(path):
            continue
        with open(path, 'rb') as f:
            h.update(name.encode() + b'\0' + f.read() + b'\0')
    return h.hexdigest()[:16]


def build_fingerprint(snapshot):
    """Combined fingerprint of ``format version + source hash + parameter snapshot``."""
    payload = json.dumps(_jsonable(snapshot), sort_keys=True, ensure_ascii=False)
    param = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f'{CACHE_FORMAT_VERSION}:{source_fingerprint()}:{param}'


# ----------------------------------------------------------------------
# Reading and writing
# ----------------------------------------------------------------------

def _sha256_file(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _atomic_bytes(blob, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(blob)
    os.replace(tmp, path)


def verify(path, keys, n_traj, T):
    """Read and validate ``path``; return the data if it passes, else ``None``.

    Every array's first axis must be ``n_traj`` (per-trajectory) or ``T+1`` (per-step
    parameters shared across trajectories); and if the former has two or more dimensions,
    axis 1 must be ``T+1``. Those two conditions are enough to catch "wrong trajectory
    count" and "half-written/truncated", without hardcoding a shape table per file.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return None
    if not isinstance(data, dict) or set(data) != set(keys):
        return None
    for v in data.values():
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            return None
        if v.shape[0] not in (n_traj, T + 1):
            return None
        if v.shape[0] == n_traj and v.ndim >= 2 and v.shape[1] != T + 1:
            return None
    return data


# ----------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------

class CacheContext:
    """Cache context for one experiment run.

    Parameters
    ----------
    dir_out : str
        Output directory for this run (keeps the existing ``data/...`` naming; directory
        names are **not** changed).
    snapshot : dict
        Every parameter that determines the output content. **The caller is responsible
        for listing them all** -- especially in the NLink case, where constants are bound
        into lambda default arguments (``dt`` / ``gamma_0`` etc. are not attributes of
        ``abg``), so they must be listed explicitly or changing them will not invalidate.
    force : bool
        If true, recompute everything (the manifest is still updated on write).
    """

    def __init__(self, dir_out, snapshot, force=False):
        self.dir_out = dir_out
        self.force = force
        self.fingerprint = build_fingerprint(snapshot)
        self._manifest = self._read_manifest()
        self._entries = dict(self._manifest.get('files', {})) if self._manifest else {}
        self.reused = 0
        self.recomputed = 0
        self._reasons = []

    # -- manifest ------------------------------------------------------

    def _manifest_path(self):
        return os.path.join(self.dir_out, MANIFEST_NAME)

    def _read_manifest(self):
        path = self._manifest_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding='utf-8') as f:
                m = json.load(f)
        except Exception:
            return None
        return m if isinstance(m, dict) else None

    # -- main interface ------------------------------------------------

    def _stale_reason(self, name, path, keys, n_traj, T, deps):
        if self.force:
            return '--force'
        if self._manifest is None:
            return 'no manifest (first run, or legacy data)'
        if self._manifest.get('version') != CACHE_FORMAT_VERSION:
            return 'cache format version changed'
        if self._manifest.get('fingerprint') != self.fingerprint:
            return 'parameters or code changed (fingerprint mismatch)'
        entry = self._entries.get(name)
        if entry is None:
            return 'no such entry in the manifest'
        for dep in deps:
            cur = _sha256_file(os.path.join(self.dir_out, dep))
            if cur is None:
                return f'dependency {dep} is missing'
            if entry.get('deps', {}).get(dep) != cur:
                return f'dependency {dep} changed'
        if verify(path, keys, n_traj, T) is None:
            return f'content validation failed (expected {n_traj} trajectories / T+1={T + 1})'
        return None

    def load(self, name, keys, n_traj, T, deps=()):
        """Return the data if it is fresh, else ``None`` (recording the reason)."""
        path = os.path.join(self.dir_out, name)
        reason = self._stale_reason(name, path, keys, n_traj, T, deps)
        if reason is not None:
            self.recomputed += 1
            self._reasons.append(f'{name}: {reason}')
            return None
        self.reused += 1
        return verify(path, keys, n_traj, T)

    def save(self, data, name, deps=()):
        """Write atomically, recording the fingerprint and each dependency's sha256 in the manifest."""
        os.makedirs(self.dir_out, exist_ok=True)
        _atomic_bytes(pickle.dumps(data), os.path.join(self.dir_out, name))
        self._entries[name] = {
            'keys': sorted(data),
            'deps': {d: _sha256_file(os.path.join(self.dir_out, d)) for d in deps},
        }
        payload = {
            'version': CACHE_FORMAT_VERSION,
            'fingerprint': self.fingerprint,
            'files': self._entries,
        }
        _atomic_bytes(json.dumps(payload, ensure_ascii=False, indent=1).encode(),
                      self._manifest_path())

    # -- reporting -----------------------------------------------------

    def summary(self):
        s = f'cache: recomputed {self.recomputed}, reused {self.reused}'
        if self._reasons:
            s += '\n' + '\n'.join(f'  - {r}' for r in self._reasons)
        return s
