# planet_pattern/physics.py
"""
Физика Живого: E = A × R × L − S

A — внимание (интенсивность сигнала)
R — резонанс (коэффициент когерентности)
L — любовь (корреляция с дыханием)
S — шум (энтропия спектра)
"""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy


def calculate_attention(signal):
    """A — внимание: интенсивность сигнала"""
    return float(np.abs(signal).mean())


def calculate_resonance(signal, fps=1.0, target_hz=0.1, band=0.03):
    """R — резонанс: коэффициент когерентности в полосе 0.1 Гц"""
    from resonance import coherence_score
    return coherence_score(signal, fps=fps, target_hz=target_hz, band=band) / 100.0


def calculate_love(signal, reference_wave):
    """L — любовь: корреляция с дыханием (эталонной волной)"""
    if len(signal) != len(reference_wave):
        min_len = min(len(signal), len(reference_wave))
        signal = signal[:min_len]
        reference_wave = reference_wave[:min_len]
    
    correlation = np.corrcoef(signal, reference_wave)[0, 1]
    if np.isnan(correlation):
        return 0.0
    # Нормализуем [-1, 1] → [0, 1]
    return float((correlation + 1.0) / 2.0)


def calculate_entropy(signal):
    """S — шум: энтропия спектра"""
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    if len(x) < 8 or np.allclose(x.std(), 0):
        return 1.0  # максимальная энтропия при отсутствии сигнала
    
    spec = np.abs(rfft(x))**2
    spec = spec / (spec.sum() + 1e-9)  # нормализация до вероятностей
    
    # Энтропия спектра (больше энтропии = больше шума)
    ent = entropy(spec + 1e-9)
    # Нормализуем на максимальную энтропию (равномерное распределение)
    max_ent = np.log(len(spec))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def calculate_energy(signal, reference_wave=None, fps=1.0):
    """
    E = A × R × L − S
    
    Возвращает словарь с компонентами и итоговой энергией.
    """
    A = calculate_attention(signal)
    R = calculate_resonance(signal, fps=fps)
    
    if reference_wave is None:
        # Если нет эталонной волны, используем чистую 0.1 Гц
        from rhythm import BreathClock
        clock = BreathClock()
        reference_wave = clock.target_wave(len(signal), fps=fps)
    
    L = calculate_love(signal, reference_wave)
    S = calculate_entropy(signal)
    
    E = A * R * L - S
    
    return {
        "A": A,
        "R": R,
        "L": L,
        "S": S,
        "E": E,
    }

