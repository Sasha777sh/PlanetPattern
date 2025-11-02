# planet_pattern/resonance.py
import numpy as np
from scipy.fft import rfft, rfftfreq


def coherence_score(signal, fps=1.0, target_hz=0.1, band=0.03):
    """
    Оцениваем «когерентность» как долю спектральной энергии в полосе вокруг 0.1 Гц.
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    if len(x) < 8 or np.allclose(x.std(), 0):
        return 0.0
    spec = np.abs(rfft(x))**2
    freqs = rfftfreq(len(x), d=1.0/fps)
    mask = (freqs >= target_hz - band) & (freqs <= target_hz + band)
    band_energy = spec[mask].sum()
    total = spec.sum() + 1e-9
    return float(100.0 * band_energy / total)  # в процентах

