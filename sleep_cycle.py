# planet_pattern/sleep_cycle.py
import numpy as np


def consolidate(centroids):
    """
    «Сон»: сворачиваем k ядер памяти в одно «ядро опыта».
    Возвращаем вектор, который можно использовать для настройки агентов.
    """
    if not centroids:
        return None
    mat = np.stack(centroids, axis=0)  # k x d
    core = mat.mean(axis=0)            # усреднение как грубая обратимость
    # нормализация
    norm = np.linalg.norm(core) + 1e-9
    return core / norm

