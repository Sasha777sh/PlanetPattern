import streamlit as st
import numpy as np
from scipy.signal import correlate
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy

st.set_page_config(page_title="Planet Pattern v2", layout="wide")

st.title("🌍 Planet Pattern — Живой vs Механический")

# Настройки
st.sidebar.header("⚙️ Параметры ритма")
freq = st.sidebar.slider("Частота дыхания (Гц)", 0.05, 0.5, 0.1, 0.01)
noise_level = st.sidebar.slider("Уровень шума", 0.0, 0.5, 0.05, 0.01)
alpha_live = st.sidebar.slider("Alpha живого агента", 0.1, 1.0, 0.5, 0.05)
alpha_fixed = st.sidebar.slider("Alpha механического", 0.1, 1.0, 0.5, 0.05)

compare_agents = st.sidebar.checkbox("Сравнить два агента", value=True)

# Генерация сигналов для двух агентов
t = np.linspace(0, 10, 500)
signal = np.sin(2 * np.pi * freq * t)
noise_live = np.random.normal(0, noise_level, len(t))
noise_fixed = np.random.normal(0, noise_level, len(t))

# Живой агент (адаптивный alpha)
mixed_live = alpha_live * signal + (1 - alpha_live) * noise_live

# Механический агент (фиксированный alpha)
mixed_fixed = alpha_fixed * signal + (1 - alpha_fixed) * noise_fixed

# Для сравнения
mixed = mixed_live if not compare_agents else mixed_live

# Эталон дыхания
ref = np.sin(2 * np.pi * 0.1 * t)

# FFT (резонанс)
fft_vals = np.abs(rfft(mixed))
fft_freq = rfftfreq(len(t), t[1] - t[0])
band_mask = (fft_freq > 0.09) & (fft_freq < 0.11)
resonance = np.sum(fft_vals[band_mask]) / (np.sum(fft_vals) + 1e-9)

# Корреляция (любовь)
correlation = np.corrcoef(mixed, ref)[0, 1]
love = (correlation + 1.0) / 2.0 if not np.isnan(correlation) else 0.0

# Внимание
attention = np.mean(np.abs(mixed))

# Шум (энтропия)
spec = np.abs(fft_vals)
spec = spec / (np.sum(spec) + 1e-9)
s = entropy(spec + 1e-9)

# Энергия жизни
E = attention * resonance * love - s

# Графики
if compare_agents:
    st.subheader("💓 Сравнение агентов")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**🌱 Живой агент (GaiaLink)**")
        st.line_chart({
            "Идеальный ритм": signal[:100],
            "Живой сигнал": mixed_live[:100],
            "Эталон": ref[:100]
        })
    
    with col_chart2:
        st.markdown("**⚙️ Механический агент (Mechanic)**")
        st.line_chart({
            "Идеальный ритм": signal[:100],
            "Механический сигнал": mixed_fixed[:100],
            "Эталон": ref[:100]
        })
else:
    col_main, col_metrics = st.columns([2, 1])
    with col_main:
        st.subheader("💓 Волна агента")
        st.line_chart({
            "Идеальный ритм": signal[:100],
            "Смешанный сигнал": mixed[:100],
            "Эталон": ref[:100]
        })

if compare_agents:
    # Расчёт для обоих агентов
    def calc_energy(signal_arr, ref_arr):
        fft_vals = np.abs(rfft(signal_arr))
        fft_freq = rfftfreq(len(signal_arr), t[1] - t[0])
        band_mask = (fft_freq > 0.09) & (fft_freq < 0.11)
        R = np.sum(fft_vals[band_mask]) / (np.sum(fft_vals) + 1e-9)
        corr = np.corrcoef(signal_arr, ref_arr)[0, 1]
        L = (corr + 1.0) / 2.0 if not np.isnan(corr) else 0.0
        A = np.mean(np.abs(signal_arr))
        spec = np.abs(fft_vals)
        spec = spec / (np.sum(spec) + 1e-9)
        S = entropy(spec + 1e-9)
        return {"A": A, "R": R, "L": L, "S": S, "E": A * R * L - S}
    
    energy_live = calc_energy(mixed_live, ref)
    energy_fixed = calc_energy(mixed_fixed, ref)
    
    # Метрики сравнения
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌱 Живой агент")
        st.metric("Энергия (E)", f"{energy_live['E']:.3f}", 
                 delta=f"{energy_live['E'] - energy_fixed['E']:.3f}")
        st.metric("Резонанс (R)", f"{energy_live['R']:.3f}")
        st.metric("Любовь (L)", f"{energy_live['L']:.3f}")
        
        if energy_live['E'] > -0.4:
            st.success("✅ В резонансе")
        elif energy_live['E'] > -0.7:
            st.warning("🌀 Потеря связи")
        else:
            st.error("⚠️ Хаос")
    
    with col2:
        st.markdown("### ⚙️ Механический агент")
        st.metric("Энергия (E)", f"{energy_fixed['E']:.3f}")
        st.metric("Резонанс (R)", f"{energy_fixed['R']:.3f}")
        st.metric("Любовь (L)", f"{energy_fixed['L']:.3f}")
        
        if energy_fixed['E'] > -0.4:
            st.success("✅ Стабилен")
        elif energy_fixed['E'] > -0.7:
            st.warning("🌀 Слабая связь")
        else:
            st.error("⚠️ Хаос")
    
    # Разница
    st.markdown("---")
    diff_E = energy_live['E'] - energy_fixed['E']
    st.metric("**Разница энергии**", f"{diff_E:+.3f}", 
             delta="Живой лучше" if diff_E > 0 else "Механический лучше")
    
else:
    with col_metrics:
        st.subheader("📊 Метрики")
        st.metric("Резонанс (R)", f"{resonance:.3f}")
        st.metric("Любовь (L)", f"{love:.3f}")
        st.metric("Внимание (A)", f"{attention:.3f}")
        st.metric("Шум (S)", f"{s:.3f}")

    # Энергия
    st.markdown("---")
    col_e1, col_e2 = st.columns([1, 3])

    with col_e1:
        st.metric("⚡ Энергия (E)", f"{E:.3f}", 
                  delta=f"A×R×L-S = {attention*resonance*love:.3f} - {s:.3f}")

    with col_e2:
        # Состояние агента
        if E > -0.4:
            st.success("✅ **Агент в резонансе:** дыхание согласовано с полем.")
        elif E > -0.7:
            st.warning("🌀 **Потеря связи с ритмом.** Возвращаюсь в дыхание…")
        else:
            st.error("⚠️ **Хаос.** Агент теряет когерентность.")

# Спектр
st.subheader("🌊 Спектр сигнала (FFT)")
spectrum_data = {
    "Частота (Гц)": fft_freq[:100],
    "Энергия": fft_vals[:100]
}
st.line_chart(spectrum_data)

st.markdown("---")
st.caption("**Формула: E = A × R × L − S** — живая энергия системы.")

