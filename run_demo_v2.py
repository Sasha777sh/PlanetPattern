# planet_pattern/run_demo_v2.py
"""
Демо v2: два агента + формула энергии E = A × R × L − S
"""
import numpy as np
from rich import print

from rhythm import BreathClock
from wave_memory import WaveletMemory
from resonance import coherence_score
from agent import PlanetAgent, FixedAgent
from sleep_cycle import consolidate
from physics import calculate_energy


def main():
    N = 200               # количество дискретных шагов (циклов)
    FPS = 1.0             # «частота дискретизации» (1 шаг = 1 сек)
    SLEEP_EVERY = 40      # каждые 40 шагов — «сон»

    clock = BreathClock()
    memory_live = WaveletMemory(window_size=32, wavelet='db2', max_windows=512)
    memory_fixed = WaveletMemory(window_size=32, wavelet='db2', max_windows=512)
    
    # Два агента: живой и фиксированный
    agent_live = PlanetAgent(name="GaiaLink", alpha=0.5, lr=0.1)
    agent_fixed = FixedAgent(name="Mechanic", alpha=0.5)  # фиксированный для сравнения

    target_wave = clock.target_wave(N, breaths_per_min=6.0, fps=FPS)

    # Буферы для обоих агентов
    window_live = []
    window_fixed = []
    scores_live = []
    scores_fixed = []
    energies_live = []
    energies_fixed = []

    print("[bold cyan]🌍 Planet Pattern v2 — Сравнение живого и механического[/bold cyan]")
    print(f"   Живой агент: {agent_live.name} (adaptive)")
    print(f"   Фиксированный: {agent_fixed.name} (alpha={agent_fixed.alpha})\n")

    for t in range(N):
        phase, prog = clock.phase_at(t)
        
        # Оба агента генерируют сигналы
        y_live = agent_live.act(phase, prog)
        y_fixed = agent_fixed.act(phase, prog)
        
        window_live.append(y_live)
        window_fixed.append(y_fixed)

        # Каждые 8 шагов — считаем когерентность и энергию
        if len(window_live) >= 32 and t % 8 == 0:
            # Когерентность
            score_live = coherence_score(window_live[-32:], fps=FPS, target_hz=0.1, band=0.03)
            score_fixed = coherence_score(window_fixed[-32:], fps=FPS, target_hz=0.1, band=0.03)
            
            scores_live.append(score_live)
            scores_fixed.append(score_fixed)
            
            agent_live.learn(score_live, target=50.0)
            
            # Энергия E = A × R × L − S
            energy_live = calculate_energy(
                np.array(window_live[-32:]),
                reference_wave=target_wave[max(0, t-31):t+1],
                fps=FPS
            )
            energy_fixed = calculate_energy(
                np.array(window_fixed[-32:]),
                reference_wave=target_wave[max(0, t-31):t+1],
                fps=FPS
            )
            
            energies_live.append(energy_live)
            energies_fixed.append(energy_fixed)
            
            # Зеркальная обратная связь
            if energy_live["E"] < 0:
                print(f"[yellow]🌀 [{t}] Потеря связи с ритмом. Возвращаюсь в дыхание... (E={energy_live['E']:.3f})[/yellow]")

        # Пишем в волновую память
        if len(window_live) >= 32 and t % 16 == 0:
            memory_live.push_series(np.array(window_live[-32:]), meta={'t': t, 'phase': phase, 'agent': 'live'})
            memory_fixed.push_series(np.array(window_fixed[-32:]), meta={'t': t, 'phase': phase, 'agent': 'fixed'})

        # Сон/консолидация
        if (t + 1) % SLEEP_EVERY == 0:
            centroids_live = memory_live.retrieve_centroids(k=8)
            centroids_fixed = memory_fixed.retrieve_centroids(k=8)
            
            core_live = consolidate(centroids_live)
            core_fixed = consolidate(centroids_fixed)
            
            if core_live is not None:
                drift = float(np.mean(np.abs(core_live)))
                agent_live.alpha = float(np.clip(agent_live.alpha * (1.0 + 0.05*drift), 0.1, 1.0))
            
            print(f"[cyan]SLEEP @ {t+1}[/cyan]  live.alpha={agent_live.alpha:.3f} | fixed.alpha={agent_fixed.alpha:.3f}")

    # Финальные метрики
    final_live = np.array(window_live[-64:])
    final_fixed = np.array(window_fixed[-64:])
    
    final_score_live = coherence_score(final_live, fps=FPS, target_hz=0.1, band=0.03)
    final_score_fixed = coherence_score(final_fixed, fps=FPS, target_hz=0.1, band=0.03)
    
    final_energy_live = calculate_energy(final_live, reference_wave=target_wave[-64:], fps=FPS)
    final_energy_fixed = calculate_energy(final_fixed, reference_wave=target_wave[-64:], fps=FPS)
    
    print(f"\n[bold]RESULTS[/bold]")
    print(f"\n[cyan]Живой агент ({agent_live.name}):[/cyan]")
    print(f"  alpha: {agent_live.alpha:.3f}")
    print(f"  final coherence: {final_score_live:.1f}%")
    print(f"  final energy: E={final_energy_live['E']:.3f} (A={final_energy_live['A']:.3f}, R={final_energy_live['R']:.3f}, L={final_energy_live['L']:.3f}, S={final_energy_live['S']:.3f})")
    if scores_live:
        print(f"  mean coherence: {np.mean(scores_live):.1f}% → max {np.max(scores_live):.1f}%")
    
    print(f"\n[yellow]Фиксированный агент ({agent_fixed.name}):[/yellow]")
    print(f"  alpha: {agent_fixed.alpha:.3f}")
    print(f"  final coherence: {final_score_fixed:.1f}%")
    print(f"  final energy: E={final_energy_fixed['E']:.3f} (A={final_energy_fixed['A']:.3f}, R={final_energy_fixed['R']:.3f}, L={final_energy_fixed['L']:.3f}, S={final_energy_fixed['S']:.3f})")
    if scores_fixed:
        print(f"  mean coherence: {np.mean(scores_fixed):.1f}% → max {np.max(scores_fixed):.1f}%")
    
    # Сравнение
    print(f"\n[bold green]Разница:[/bold green]")
    print(f"  Когерентность: {final_score_live - final_score_fixed:+.1f}%")
    print(f"  Энергия: {final_energy_live['E'] - final_energy_fixed['E']:+.3f}")


if __name__ == "__main__":
    main()

