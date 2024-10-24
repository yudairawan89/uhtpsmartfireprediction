# Import Streamlit dan pustaka yang diperlukan
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit.runtime.legacy_caching import caching
from streamlit_autorefresh import st_autorefresh

# Menambahkan logo di sebelah kiri tulisan "UHTP Smart Fire Prediction"
col1, col2 = st.columns([1, 6])  # Membuat layout kolom untuk logo dan judul
with col1:
    st.image("logo.png", width=100)  # Menambahkan logo dari folder yang sama dengan aplikasi
with col2:
    # Judul Aplikasi
    st.title("UHTP Smart Fire Prediction")

# Fungsi untuk memuat data
@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Fungsi untuk memuat scaler yang sudah dilatih
@st.cache_resource
def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat scaler: {e}")
        return None

# URL Data Google Sheets (format CSV)
data_url = 'https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv'

# Refresh otomatis setiap 3 detik
st_autorefresh(interval=3000, key="data_refresh")

# Tombol untuk refresh data manual
if st.button('Refresh Data'):
    caching.clear_cache()  # Hapus cache agar data terbaru dimuat

# Deskripsi aplikasi yang dipindahkan ke bawah tombol
st.markdown("""
Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan menggunakan pengembangan model Hybrid Machine dan Deep Learning.
Data diambil dari perangkat IoT secara Realtime [Google Sheets](https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?usp=sharing).
""")

# Muat Data
sensor_data = load_data(data_url)

if sensor_data is not None:
    st.subheader("Data Sensor")
    st.dataframe(sensor_data)

    # Mengganti nama kolom sesuai dengan model yang dilatih
    sensor_data = sensor_data.rename(columns={
        'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
        'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Perbukaan Tanah'
    })

    # Muat Model dan Scaler
    model = load_model('meta_LR.joblib')
    scaler = load_scaler('scaler.joblib')

    if model is not None and scaler is not None:
        # Pra-pemrosesan Data
        fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)',
                 'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Perbukaan Tanah']

        if all(col in sensor_data.columns for col in fitur):
            fitur_data = sensor_data[fitur]

            # Mengganti koma dengan titik agar bisa dikonversi ke float
            for col in fitur_data.columns:
                fitur_data[col] = fitur_data[col].astype(str).str.replace(',', '.').astype(float)

            # Mengisi nilai yang hilang dengan 0
            fitur_data.fillna(0, inplace=True)

            # Standarisasi Fitur menggunakan scaler yang sudah dilatih
            fitur_scaled = scaler.transform(fitur_data)

            # Buat DataFrame dari fitur yang sudah di-scale
            fitur_scaled_df = pd.DataFrame(fitur_scaled, columns=fitur)

            # Prediksi
            predictions = model.predict(fitur_scaled_df)

            # Konversi prediksi numerik ke label kategori
            def convert_to_label(pred):
                if pred == 0:
                    return "High"
                elif pred == 1:
                    return "Low"
                elif pred == 2:
                    return "Moderate"
                elif pred == 3:
                    return "Very High"
                else:
                    return "Unknown"

            sensor_data['Prediksi Kebakaran'] = [convert_to_label(pred) for pred in predictions]

            # Menambahkan tampilan hasil prediksi di bawah data sensor
            st.subheader("Hasil Prediksi")
            st.dataframe(sensor_data)

            # Fitur download hasil prediksi sebagai CSV
            csv = sensor_data.to_csv(index=False)
            st.download_button(
                label="Download Hasil Prediksi sebagai CSV",
                data=csv,
                file_name='hasil_prediksi_kebakaran.csv',
                mime='text/csv'
            )

            # Tampilkan hasil prediksi data paling akhir setelah sensor data
            st.subheader("Hasil Prediksi Data Paling Akhir")
            with st.expander("Klik untuk melihat detail variabel dan hasil prediksi"):
                last_row = sensor_data.iloc[-1]
                st.write("**Variabel Data Paling Akhir:**")
                st.write(last_row[fitur])

                # Prediksi Kebakaran berdasarkan risiko dengan warna dan latar belakang
                risk = last_row['Prediksi Kebakaran']
                risk_styles = {
                    "Low": {"color": "white", "background-color": "blue"},
                    "Moderate": {"color": "white", "background-color": "green"},
                    "High": {"color": "black", "background-color": "yellow"},
                    "Very High": {"color": "white", "background-color": "red"}
                }

                risk_style = risk_styles.get(risk, {"color": "black", "background-color": "white"})

                st.markdown(
                    f"<p style='color:{risk_style['color']}; background-color:{risk_style['background-color']}; font-weight: bold; padding: 10px; border-radius: 5px;'>Prediksi Kebakaran: {risk}</p>", 
                    unsafe_allow_html=True
                )

            # Fitur Input Manual untuk Prediksi Real-time
            st.subheader("Prediksi Kebakaran Baru")
            st.markdown("Masukkan nilai sensor untuk memprediksi kemungkinan kebakaran.")

            suhu = st.number_input("Suhu Udara (°C)", min_value=0.0, max_value=100.0, value=25.0)
            kelembapan_udara = st.number_input("Kelembapan Udara (%)", min_value=0.0, max_value=100.0, value=50.0)
            curah_hujan = st.number_input("Curah Hujan/Jam (mm)", min_value=0.0, max_value=500.0, value=10.0)
            kecepatan_angin = st.number_input("Kecepatan Angin (ms)", min_value=0.0, max_value=100.0, value=5.0)
            kelembapan_tanah = st.number_input("Kelembapan Tanah (%)", min_value=0.0, max_value=100.0, value=40.0)

            # Buat DataFrame dari input pengguna
            input_data = pd.DataFrame({
                'Tavg: Temperatur rata-rata (°C)': [suhu],
                'RH_avg: Kelembapan rata-rata (%)': [kelembapan_udara],
                'RR: Curah hujan (mm)': [curah_hujan],
                'ff_avg: Kecepatan angin rata-rata (m/s)': [kecepatan_angin],
                'Kelembaban Perbukaan Tanah': [kelembapan_tanah]
            })

            # Pra-pemrosesan input pengguna menggunakan scaler yang sudah dilatih
            input_scaled = scaler.transform(input_data)

            # Prediksi untuk input pengguna
            user_prediction = model.predict(input_scaled)
            user_label = convert_to_label(user_prediction[0])

            # Menampilkan hasil prediksi dengan background warna
            user_risk_style = risk_styles.get(user_label, {"color": "black", "background-color": "white"})

            st.markdown(
                f"<p style='color:{user_risk_style['color']}; background-color:{user_risk_style['background-color']}; font-weight: bold; padding: 10px; border-radius: 5px;'>Prediksi Risiko Kebakaran: {user_label}</p>", 
                unsafe_allow_html=True
            )
