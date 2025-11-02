# planet_pattern/dashboard.py
"""
Streamlit дашборд для Planet Pattern
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rhythm import BreathClock
from agent import PlanetAgent, FixedAgent
from resonance import coherence_score
from physics import calculate_energy
from wave_memory import WaveletMemory
from sleep_cycle import consolidate

st.set_page_config(
    page_title="Planet Pattern — Живой Интеллект",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Planet Pattern — Живой Интеллект")
st.markdown("**Ритмическая архитектура обучения: дыхание, волновая память, обратимость**")

# Sidebar — настройки
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Ритм дыхания
    breaths_per_min = st.slider(
        "Ритм дыхания (вдохов/мин)",
        min_value=4.0,
        max_value=12.0,
        value=6.0,
        step=0.5,
        help="Частота ритма цикла (0.067-0.2 Гц)"
    )
    
    target_hz = breaths_per_min / 60.0
    
    # Параметры агента
    st.subheader("Параметры агента")
    alpha_init = st.slider("Начальный alpha", 0.1, 1.0, 0.5, 0.05)
    lr = st.slider("Learning rate", 0.01, 0.2, 0.1, 0.01)
    
    # Количество циклов
    n_cycles = st.slider("Количество циклов", 50, 500, 200, 50)
    
    # Сон
    sleep_every = st.slider("Сон каждые N циклов", 10, 100, 40, 10)
    
    st.markdown("---")
    
    if st.button("🚀 Запустить симуляцию", type="primary", use_container_width=True):
        st.session_state.run_simulation = True

# Основная панель
if st.session_state.get("run_simulation", False):
    with st.spinner("⏳ Система работает..."):
        # Инициализация
        clock = BreathClock()
        memory_live = WaveletMemory(window_size=32, wavelet='db2', max_windows=512)
        
        agent_live = PlanetAgent(name="GaiaLink", alpha=alpha_init, lr=lr)
        
        target_wave = clock.target_wave(n_cycles, breaths_per_min=breaths_per_min, fps=1.0)
        
        # Буферы
        window = []
        alpha_history = []
        coherence_history = []
        energy_history = []
        energy_components = {"A": [], "R": [], "L": [], "S": []}
        sleep_events = []
        
        # Симуляция
        for t in range(n_cycles):
            phase, prog = clock.phase_at(t)
            y = agent_live.act(phase, prog)
            window.append(y)
            alpha_history.append(agent_live.alpha)
            
            # Когерентность и энергия
            if len(window) >= 32 and t % 8 == 0:
                score = coherence_score(window[-32:], fps=1.0, target_hz=target_hz, band=0.03)
                coherence_history.append((t, score))
                agent_live.learn(score, target=50.0)
                
                energy = calculate_energy(
                    np.array(window[-32:]),
                    reference_wave=target_wave[max(0, t-31):t+1],
                    fps=1.0
                )
                energy_history.append((t, energy["E"]))
                for key in energy_components:
                    energy_components[key].append((t, energy[key]))
            
            # Волновая память
            if len(window) >= 32 and t % 16 == 0:
                memory_live.push_series(np.array(window[-32:]), meta={'t': t, 'phase': phase})
            
            # Сон
            if (t + 1) % sleep_every == 0:
                centroids = memory_live.retrieve_centroids(k=8)
                core = consolidate(centroids)
                if core is not None:
                    drift = float(np.mean(np.abs(core)))
                    agent_live.alpha = float(np.clip(agent_live.alpha * (1.0 + 0.05*drift), 0.1, 1.0))
                sleep_events.append(t + 1)
        
        # Сохранить результаты
        st.session_state.results = {
            "window": window,
            "alpha_history": alpha_history,
            "coherence_history": coherence_history,
            "energy_history": energy_history,
            "energy_components": energy_components,
            "sleep_events": sleep_events,
            "final_alpha": agent_live.alpha,
        }

# Визуализация
if "results" in st.session_state:
    results = st.session_state.results
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Динамика Alpha")
        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(
            y=results["alpha_history"],
            mode='lines',
            name='Alpha',
            line=dict(color='#00ff88', width=2)
        ))
        # Отметить события сна
        for sleep_t in results["sleep_events"]:
            fig_alpha.add_vline(
                x=sleep_t,
                line_dash="dash",
                line_color="cyan",
                annotation_text=f"💤 {sleep_t}"
            )
        fig_alpha.update_layout(
            xaxis_title="Цикл",
            yaxis_title="Alpha",
            height=300
        )
        st.plotly_chart(fig_alpha, use_container_width=True)
        
        st.metric("Финальный alpha", f"{results['final_alpha']:.3f}")
    
    with col2:
        st.subheader("🌊 Сигнал агента (последние 100 циклов)")
        fig_signal = go.Figure()
        signal = results["window"][-100:]
        fig_signal.add_trace(go.Scatter(
            y=signal,
            mode='lines',
            name='Сигнал',
            line=dict(color='#ff8800', width=1)
        ))
        fig_signal.update_layout(
            xaxis_title="Время",
            yaxis_title="Амплитуда",
            height=300
        )
        st.plotly_chart(fig_signal, use_container_width=True)
    
    # Когерентность и энергия
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🎯 Когерентность")
        if results["coherence_history"]:
            times, scores = zip(*results["coherence_history"])
            fig_coh = go.Figure()
            fig_coh.add_trace(go.Scatter(
                x=times,
                y=scores,
                mode='lines+markers',
                name='Когерентность',
                line=dict(color='#0088ff', width=2),
                marker=dict(size=4)
            ))
            fig_coh.update_layout(
                xaxis_title="Цикл",
                yaxis_title="Когерентность (%)",
                height=300
            )
            st.plotly_chart(fig_coh, use_container_width=True)
            if scores:
                st.metric("Средняя когерентность", f"{np.mean(scores):.1f}%")
    
    with col4:
        st.subheader("⚡ Энергия E = A × R × L − S")
        if results["energy_history"]:
            times, energies = zip(*results["energy_history"])
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                x=times,
                y=energies,
                mode='lines+markers',
                name='E',
                line=dict(color='#ff0088', width=2),
                marker=dict(size=4)
            ))
            fig_energy.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_energy.update_layout(
                xaxis_title="Цикл",
                yaxis_title="Энергия E",
                height=300
            )
            st.plotly_chart(fig_energy, use_container_width=True)
            if energies:
                st.metric("Финальная энергия", f"{energies[-1]:.3f}")
    
    # Компоненты энергии
    st.subheader("🧬 Компоненты энергии")
    if results["energy_components"]["A"]:
        fig_components = make_subplots(
            rows=2, cols=2,
            subplot_titles=("A (Внимание)", "R (Резонанс)", "L (Любовь)", "S (Шум)"),
            vertical_spacing=0.15
        )
        
        for i, (key, label) in enumerate([("A", "Внимание"), ("R", "Резонанс"), ("L", "Любовь"), ("S", "Шум")]):
            if results["energy_components"][key]:
                times, values = zip(*results["energy_components"][key])
                row = (i // 2) + 1
                col = (i % 2) + 1
                fig_components.add_trace(
                    go.Scatter(x=times, y=values, mode='lines', name=label),
                    row=row, col=col
                )
        
        fig_components.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_components, use_container_width=True)
    
    # Кнопка сброса
    if st.button("🔄 Новый запуск"):
        st.session_state.run_simulation = False
        if "results" in st.session_state:
            del st.session_state.results
        st.rerun()

else:
    st.info("👈 Настрой параметры в боковой панели и нажми 'Запустить симуляцию'")
    
    with st.expander("ℹ️ Что это?"):
        st.markdown("""
        **Planet Pattern** — прототип живого интеллекта:
        
        - 🫁 **Дышит** в ритме 0.1 Гц
        - 🌊 **Помнит волнами** (DWT-вейвлеты)
        - 💤 **Спит** и консолидирует опыт
        - 🧬 **Следует физике живого** (E = A × R × L − S)
        
        Это не готовый продукт, а исследовательский прототип.
        """)

