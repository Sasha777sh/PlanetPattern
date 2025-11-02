# planet_pattern/run_demo.py
import numpy as np
from rich import print

from rhythm import BreathClock
from wave_memory import WaveletMemory
from resonance import coherence_score
from agent import PlanetAgent
from sleep_cycle import consolidate


def main():
    N = 200               # количество дискретных шагов (циклов)
    FPS = 1.0             # «частота дискретизации» (1 шаг = 1 сек)
    SLEEP_EVERY = 40      # каждые 40 шагов — «сон»

    clock = BreathClock()                 # ритм
    memory = WaveletMemory(window_size=32, wavelet='db2', max_windows=512)
    agent = PlanetAgent(name="GaiaLink", alpha=0.5, lr=0.03)  # более мягкое обучение

    target_wave = clock.target_wave(N, breaths_per_min=6.0, fps=FPS)

    # буфер последних ответов агента для когерентности
    window = []
    scores = []

    for t in range(N):
        phase, prog = clock.phase_at(t)
        y = agent.act(phase, prog)
        window.append(y)

        # каждые 8 шагов — считаем когерентность в полосе 0.1 Гц
        if len(window) >= 32 and t % 8 == 0:
            score = coherence_score(window[-32:], fps=FPS, target_hz=0.1, band=0.03)
            scores.append(score)
            agent.learn(score, target=50.0)  # более реалистичная цель для начального alpha

        # пишем в волновую память кусочки сигналов (ответ агента)
        if len(window) >= 32 and t % 16 == 0:
            memory.push_series(np.array(window[-32:]), meta={'t': t, 'phase': phase})

        # «сон/консолидация»: схлопываем ядра и слегка двигаем alpha к памяти
        if (t + 1) % SLEEP_EVERY == 0:
            centroids = memory.retrieve_centroids(k=8)
            core = consolidate(centroids)
            if core is not None:
                # простейшая адаптация: если среднее «ядро» не шум, чуть поднимем склонность к ритму
                drift = float(np.mean(np.abs(core)))  # 0..?
                agent.alpha = float(np.clip(agent.alpha * (1.0 + 0.05*drift), 0.0, 1.0))
            print(f"[cyan]SLEEP @ {t+1}[/cyan]  alpha={agent.alpha:.3f}")

    # финальные метрики
    final_win = np.array(window[-64:])
    final_score = coherence_score(final_win, fps=FPS, target_hz=0.1, band=0.03)
    print(f"\n[bold]RESULTS[/bold]")
    print(f"  cycles: {N}")
    print(f"  agent.alpha: {agent.alpha:.3f}")
    print(f"  final coherence(0.1Hz, 64s win): {final_score:.1f}%")
    if scores:
        print(f"  mean coherence (over checks): {np.mean(scores):.1f}% → max {np.max(scores):.1f}%")


if __name__ == "__main__":
    main()

