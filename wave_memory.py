# planet_pattern/wave_memory.py
import numpy as np
import pywt
from collections import deque


class WaveletMemory:
    """
    Память-волна: мы накапливаем окна сигналов (массивы чисел),
    сворачиваем Discrete Wavelet Transform (DWT) → хранится компактный «слой».
    """
    def __init__(self, window_size=32, wavelet='db2', max_windows=256):
        self.window_size = window_size
        self.wavelet = wavelet
        self.max_windows = max_windows
        self.buffer = deque(maxlen=max_windows)   # список (coeffs, meta)

    def push_series(self, series, meta=None):
        """
        series: 1D массив длины >= window_size
        режем на окна, каждое окно → DWT коэффициенты → в память
        """
        series = np.asarray(series, dtype=float)
        if len(series) < self.window_size:
            return 0
        count = 0
        for i in range(0, len(series) - self.window_size + 1, self.window_size):
            win = series[i:i+self.window_size]
            coeffs = pywt.wavedec(win, self.wavelet, level=None, mode='symmetric')
            packed = np.concatenate([c.flatten() for c in coeffs])
            self.buffer.append((packed, meta))
            count += 1
        return count

    def retrieve_centroids(self, k=8):
        """
        Грубая «консолидация»: берём k равномерных «ядёр» из памяти.
        (Можно заменить на KMeans, но сохраняем зависимости минимальными)
        """
        if not self.buffer:
            return []
        step = max(1, len(self.buffer) // k)
        return [self.buffer[i][0] for i in range(0, len(self.buffer), step)][:k]

