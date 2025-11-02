# planet_pattern/rhythm.py
import numpy as np


class BreathClock:
    """
    Дискретный «пульс»: вдох→пауза→выдох→тишина.
    Не ждём реального времени — считаем фазу по индексу цикла.
    """
    def __init__(self, inhale=4, hold1=2, exhale=6, hold2=2):
        self.pattern = [('inhale', inhale), ('hold1', hold1),
                        ('exhale', exhale), ('hold2', hold2)]
        self.period = sum(d for _, d in self.pattern)

    def phase_at(self, t):
        """Возвращает фазу и локальный прогресс в рамках периода."""
        t_mod = t % self.period
        acc = 0
        for name, dur in self.pattern:
            if acc <= t_mod < acc + dur:
                # нормированный прогресс в текущей фазе [0..1]
                return name, (t_mod - acc) / max(1, dur)
            acc += dur
        return 'hold2', 1.0

    def target_wave(self, length, breaths_per_min=6.0, fps=1.0):
        """
        Синтетическая целевая волна 0.1 Гц (для расчёта резонанса).
        length — число дискретов, fps — «сэмплов в секунду»
        """
        f = breaths_per_min / 60.0  # 0.1 Гц
        t = np.arange(length) / fps
        return np.sin(2 * np.pi * f * t)

