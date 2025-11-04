# planet_pattern/llm_resonance.py
"""
Резонансный слой для LLM — применение Физики Живого к языковым моделям.

Интеграция Planet Pattern в LLM архитектуру:
- E = A × R × L − S для текстовых данных
- Резонанс с ритмом диалога (пульс, паузы, эмоции)
- Адаптивное обучение через энергию, а не только через loss
"""

import numpy as np
from typing import List, Dict, Optional
from physics import calculate_energy


class LLMResonanceLayer:
    """
    Резонансный слой для LLM, который измеряет "живость" диалога.
    
    Применяет формулу E = A × R × L − S к:
    - A: Attention weights (интенсивность внимания)
    - R: Resonance с ритмом диалога (0.1 Hz pattern)
    - L: Love (корреляция с эталонным "живым" ответом)
    - S: Noise (энтропия токенов, хаос)
    """
    
    def __init__(self, target_hz=0.1):
        self.target_hz = target_hz
        self.energy_history = []
        self.dialog_rhythm = []  # история временных меток токенов
        
    def calculate_attention_energy(self, attention_weights):
        """
        A (Attention) — средняя интенсивность внимания
        
        attention_weights: [n_layers, n_heads, seq_len, seq_len]
        или упрощённо: средние веса внимания по слоям
        """
        if isinstance(attention_weights, np.ndarray):
            # Если массив — берём среднее
            return float(np.abs(attention_weights).mean())
        elif isinstance(attention_weights, (list, tuple)):
            # Если список тензоров — усредняем
            weights = np.array([np.abs(w).mean() if hasattr(w, 'mean') else w for w in attention_weights])
            return float(weights.mean())
        else:
            # Fallback: если не можем посчитать — возвращаем 1.0
            return 1.0
    
    def calculate_text_resonance(self, token_times, fps=1.0):
        """
        R (Resonance) — резонанс с ритмом диалога
        
        token_times: список временных меток токенов (или индексов)
        Ищем паттерн 0.1 Hz в последовательности токенов
        """
        if len(token_times) < 8:
            return 0.5  # нейтральный резонанс для коротких последовательностей
        
        # Преобразуем временные метки в сигнал
        signal = np.array(token_times, dtype=float)
        signal = signal - signal.mean()  # центрируем
        
        if np.allclose(signal.std(), 0):
            return 0.5
        
        # FFT для поиска резонанса с 0.1 Hz
        from scipy.fft import rfft, rfftfreq
        spec = np.abs(rfft(signal))**2
        freqs = rfftfreq(len(signal), d=1.0/fps)
        
        # Ищем энергию в полосе 0.1 Hz ± 0.03
        band_mask = (freqs >= self.target_hz - 0.03) & (freqs <= self.target_hz + 0.03)
        band_energy = spec[band_mask].sum()
        total_energy = spec.sum() + 1e-9
        
        return float(band_energy / total_energy)
    
    def calculate_love(self, response_embedding, reference_embedding=None):
        """
        L (Love) — корреляция с эталонным "живым" ответом
        
        response_embedding: эмбеддинг ответа модели
        reference_embedding: эталонный эмбеддинг (например, "живой" ответ)
        
        Если нет эталона — используем норму эмбеддинга как proxy
        """
        if reference_embedding is None:
            # Если нет эталона — используем норму как proxy живости
            if isinstance(response_embedding, np.ndarray):
                norm = np.linalg.norm(response_embedding)
                # Нормализуем на разумный диапазон (обычно эмбеддинги 0-1)
                return float(np.clip(norm / np.sqrt(len(response_embedding)), 0, 1))
            else:
                return 0.5  # нейтральное значение
        
        # Корреляция с эталоном
        resp = np.asarray(response_embedding).flatten()
        ref = np.asarray(reference_embedding).flatten()
        
        if len(resp) != len(ref):
            min_len = min(len(resp), len(ref))
            resp = resp[:min_len]
            ref = ref[:min_len]
        
        correlation = np.corrcoef(resp, ref)[0, 1]
        if np.isnan(correlation):
            return 0.5
        
        # Нормализуем [-1, 1] → [0, 1]
        return float((correlation + 1.0) / 2.0)
    
    def calculate_entropy(self, token_probs):
        """
        S (Noise) — энтропия распределения токенов
        
        token_probs: вероятности токенов [batch_size, vocab_size]
        или список вероятностей
        """
        from scipy.stats import entropy
        
        if isinstance(token_probs, np.ndarray):
            if token_probs.ndim == 1:
                probs = token_probs
            else:
                # Если 2D — усредняем по batch
                probs = token_probs.mean(axis=0)
        else:
            probs = np.array(token_probs)
        
        # Нормализуем до вероятностей
        probs = probs / (probs.sum() + 1e-9)
        
        # Энтропия
        ent = entropy(probs + 1e-9)
        max_ent = np.log(len(probs))
        
        # Нормализуем на максимальную энтропию
        return float(ent / max_ent) if max_ent > 0 else 0.0
    
    def calculate_llm_energy(self, 
                          attention_weights=None,
                          token_times=None,
                          response_embedding=None,
                          token_probs=None,
                          reference_embedding=None):
        """
        Вычисляет энергию E = A × R × L − S для LLM
        
        Возвращает словарь с компонентами и итоговой энергией.
        """
        # A — внимание
        A = self.calculate_attention_energy(attention_weights) if attention_weights is not None else 0.5
        
        # R — резонанс
        if token_times is not None:
            R = self.calculate_text_resonance(token_times)
        else:
            R = 0.5  # нейтральное значение
        
        # L — любовь (корреляция с эталоном)
        L = self.calculate_love(response_embedding, reference_embedding)
        
        # S — шум (энтропия)
        S = self.calculate_entropy(token_probs) if token_probs is not None else 0.5
        
        # Энергия
        E = A * R * L - S
        
        result = {
            "A": A,
            "R": R,
            "L": L,
            "S": S,
            "E": E
        }
        
        self.energy_history.append(result)
        return result
    
    def adapt_temperature(self, energy, base_temperature=0.7):
        """
        Адаптирует temperature на основе энергии
        
        Высокая энергия → ниже temperature (более детерминированный ответ)
        Низкая энергия → выше temperature (более креативный ответ)
        """
        # Если E > 0 — высокий резонанс, уменьшаем temperature
        # Если E < 0 — низкий резонанс, увеличиваем temperature
        delta = -energy * 0.2  # масштабируем влияние
        new_temp = base_temperature + delta
        
        # Ограничиваем разумными значениями
        return float(np.clip(new_temp, 0.1, 1.5))
    
    def get_energy_feedback(self, energy_result):
        """
        Возвращает текстовую обратную связь на основе энергии
        (как в PlanetAgent — "🌀 Потеря связи с ритмом...")
        """
        E = energy_result["E"]
        
        if E > 0.3:
            return "✅ В резонансе — ответ живой и связанный"
        elif E > 0:
            return "🌀 Умеренная связь — ответ стабилен"
        elif E > -0.3:
            return "⚠️ Потеря связи — ответ теряет когерентность"
        else:
            return "❌ Хаос — ответ дезориентирован"


def integrate_with_llm(llm_output, attention_weights=None, token_probs=None):
    """
    Утилита для интеграции резонансного слоя с существующим LLM
    
    Пример использования:
        response = llm.generate(prompt)
        energy = integrate_with_llm(
            llm_output=response,
            attention_weights=llm.attention_weights,
            token_probs=llm.token_probs
        )
    """
    layer = LLMResonanceLayer()
    
    # Генерируем временные метки (простые индексы)
    token_times = list(range(len(llm_output.split())))
    
    # Вычисляем энергию
    energy = layer.calculate_llm_energy(
        attention_weights=attention_weights,
        token_times=token_times,
        token_probs=token_probs
    )
    
    return energy, layer.get_energy_feedback(energy)

