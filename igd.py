# Simulasi Antrian IGD Rumah Sakit
# Model: M/M/c (Multiple Servers)
# Framework: Streamlit

import streamlit as st

st.set_page_config(page_title="Simulasi Antrian IGD", layout="centered")

st.title("🏥 Simulasi Teori Antrian IGD Rumah Sakit")
st.caption("Model Antrian M/M/c (Multiple Servers)")

st.markdown("---")

st.header("1. Input Data")

arrival = st.number_input(
    "Waktu antar kedatangan pasien (menit)",
    min_value=0.1,
    step=0.1
)

service = st.number_input(
    "Waktu pelayanan per pasien (menit)",
    min_value=0.1,
    step=0.1
)

c = st.number_input(
    "Jumlah pelayan / dokter (c)",
    min_value=1,
    step=1,
    value=2
)

if st.button("Hitung"):
    if arrival <= 0 or service <= 0 or c <= 0:
        st.error("Input tidak valid. Semua nilai harus lebih dari 0.")
    else:
        # Perhitungan dasar
        lambda_rate = 1 / arrival
        mu = 1 / service
        rho = lambda_rate / (c * mu)

        if rho >= 1:
            st.error("Sistem tidak stabil. Laju kedatangan lebih besar dari kapasitas pelayanan.")
        else:
            # Rumus sesuai instruksi (aproksimasi)
            W = 1 / (mu - lambda_rate / c)
            Wq = (lambda_rate ** 2) / (c * mu * (mu - lambda_rate / c))

            st.success("Perhitungan berhasil")

            st.subheader("Hasil Perhitungan")
            st.write(f"**Laju kedatangan (λ):** {lambda_rate:.3f} pasien/menit")
            st.write(f"**Laju pelayanan per pelayan (μ):** {mu:.3f} pasien/menit")
            st.write(f"**Pemanfaatan pelayan (ρ):** {rho:.3f}")
            st.write(f"**Waktu rata-rata dalam sistem (W):** {W:.3f} menit")
            st.write(f"**Waktu rata-rata tunggu dalam antrian (Wq):** {Wq:.3f} menit")

            # ==============================
            # Grafik Waktu Tunggu Pasien
            # ==============================
            import matplotlib.pyplot as plt
            import numpy as np

            st.subheader("Grafik Waktu Tunggu Pasien (Wq)")

            arrival_range = np.linspace(arrival * 0.5, arrival * 1.5, 20)
            lambda_range = 1 / arrival_range

            Wq_values = []
            for l in lambda_range:
                if l < c * mu:
                    Wq_values.append((l ** 2) / (c * mu * (mu - l / c)))
                else:
                    Wq_values.append(None)

            fig, ax = plt.subplots()
            ax.plot(arrival_range, Wq_values)
            ax.set_xlabel("Waktu antar kedatangan pasien (menit)")
            ax.set_ylabel("Waktu tunggu rata-rata (menit)")
            ax.set_title("Hubungan Waktu Antar Kedatangan vs Waktu Tunggu Pasien")

            st.pyplot(fig)

st.markdown("---")

st.header("2. Ilustrasi Alur Antrian")
st.info(
    "Pasien datang ke IGD → menunggu dalam antrian → dilayani oleh dokter/perawat → keluar sistem"
)

st.markdown("---")

st.header("3. Tutorial & Penjelasan Teori Antrian")
st.markdown(
    """
**Model M/M/c** digunakan untuk menggambarkan sistem pelayanan IGD dengan:

- Kedatangan pasien mengikuti distribusi **Poisson (M)**
- Waktu pelayanan mengikuti distribusi **Eksponensial (M)**
- Terdapat **c pelayan** (dokter/perawat)

### Langkah Perhitungan
1. **Laju kedatangan (λ)** = 1 / waktu antar kedatangan
2. **Laju pelayanan per pelayan (μ)** = 1 / waktu pelayanan
3. **Pemanfaatan pelayan (ρ)** = λ / (cμ)
4. **Waktu rata-rata dalam sistem (W)** = 1 / (μ − λ/c)
5. **Waktu rata-rata tunggu dalam antrian (Wq)** = λ² / [cμ(μ − λ/c)]

Aplikasi ini digunakan sebagai **media pembelajaran dan simulasi sederhana** teori antrian.
"""
)

st.caption("Aplikasi edukasi – Simulasi Antrian IGD Rumah Sakit")
