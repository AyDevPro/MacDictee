"""mlx-whisper sans numba ni scipy : ~230 Mo de moins dans l'app, et pas de compilateur LLVM chargé en mémoire.

mlx-whisper ne s'en sert que pour l'horodatage des mots (réunions) : un alignement DTW (numba) et un filtre
médian (scipy). On les remplace par des versions numpy qui donnent les mêmes résultats.
À importer AVANT mlx_whisper.
"""
import sys
import types

import numpy as np


def medfilt(x, kernel_size):
    """scipy.signal.medfilt (bords complétés par des zéros), filtre sur le dernier axe uniquement."""
    width = kernel_size[-1] if isinstance(kernel_size, (tuple, list)) else kernel_size
    pad = width // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1)
    return np.median(windows, axis=-1).astype(x.dtype)


def dtw(x: np.ndarray) -> np.ndarray:
    """Même alignement que mlx_whisper.timing.dtw_cpu (mêmes règles en cas d'égalité), diagonale par diagonale."""
    n, m = x.shape
    x = x.astype(np.float32)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    trace = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0
    for d in range(2, n + m + 1):  # i + j = d : toutes les cases d'une anti-diagonale sont indépendantes
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        j = d - i
        c0, c1, c2 = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
        t = np.where((c0 < c1) & (c0 < c2), 0, np.where((c1 < c0) & (c1 < c2), 1, 2))
        cost[i, j] = x[i - 1, j - 1] + np.choose(t, [c0, c1, c2])
        trace[i, j] = t
    trace[0, :] = 2
    trace[:, 0] = 1
    i, j, path = n, m, []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        t = trace[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return np.array(path[::-1]).T


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


# numba : on n'en veut jamais (même installé, il chargerait LLVM) ; mlx_whisper n'utilise que @numba.jit
_stub("numba", jit=lambda *args, **kwargs: (lambda f: f))
try:
    from scipy import signal  # noqa: F401  (présent en développement)
except ImportError:  # app autonome : scipy n'est pas embarqué
    _stub("scipy.signal", medfilt=medfilt)
    _stub("scipy", signal=sys.modules["scipy.signal"])

from mlx_whisper import timing  # noqa: E402

timing.dtw = dtw
