# planet_pattern/agent.py
import numpy as np


class PlanetAgent:
    """
    Простой «агент-резонатор».
    У него есть внутренний параметр alpha — насколько он следует целевой волне 0.1 Гц.
    Обучение: если резонанс ↑ — чуть увеличиваем alpha; если ↓ — уменьшаем.
    """
    def __init__(self, name, alpha=0.5, lr=0.1, adaptive=True):
        self.name = name
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.adaptive = adaptive  # True = живой агент, False = фиксированный

    def act(self, phase, local_progress, noise_scale=0.2):
        """
        Генерируем «ответ-сигнал» цикла как смесь:
        - синуса текущей фазы (идеальный ритм)
        - шума/самовыражения
        alpha определяет долю следования ритму.
        """
        # идеальный ритм задаём синусом от прогресса фазы (0..1)
        ideal = np.sin(2*np.pi*local_progress)
        noise = np.random.normal(0, noise_scale)
        y = self.alpha * ideal + (1 - self.alpha) * noise
        return float(y)

    def learn(self, last_score, target=70.0):
        """
        Самонастройка с плавным восстановлением.
        Ищет состояние «дыхания» между хаосом и порядком.
        """
        if not self.adaptive:
            return  # фиксированный агент не учится
        
        # Нормализуем gap более мягко
        gap = (last_score - target) / 100.0
        
        # Плавное восстановление после провала
        if last_score < target / 2:
            self.alpha += 0.02  # мягкое восстановление после провала
            self.alpha = np.clip(self.alpha, 0.1, 1.0)
            return
        
        # Если score близок к target или выше → поощряем alpha
        if last_score >= target * 0.8:  # если достигли 80% от цели
            gap = abs(gap) * 0.5  # мягче уменьшаем, если уже близко
        
        self.alpha = np.clip(self.alpha + self.lr * gap, 0.1, 1.0)  # минимум 0.1


class FixedAgent(PlanetAgent):
    """
    Фиксированный агент для сравнения (механический, не живой).
    """
    def __init__(self, name, alpha=0.5):
        super().__init__(name, alpha=alpha, lr=0.0, adaptive=False)
