# Import Streamlit dan pustaka yang diperlukan
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import io

# Judul Aplikasi
st.title("Prediksi Kebakaran")

# Deskripsi Aplikasi
st.markdown("""
Aplikasi untuk memprediksi tingkat kemungkinan kebakaran berdasarkan data sensor.
Data diambil dari [Google Sheets](https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?usp=sharing).
""")

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

# URL Data Google Sheets (format CSV)
data_url = 'https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv'

# Tombol untuk refresh data
if st.button('Refresh Data'):
    st.cache_data.clear()  # Hapus cache agar data terbaru dimuat

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

    # Muat Model
    model = load_model('meta_LR.joblib')

    if model is not None:
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

            # Standarisasi Fitur (jika model membutuhkan)
            scaler = StandardScaler()
            fitur_scaled = scaler.fit_transform(fitur_data)

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

            # Tampilkan hasil prediksi data paling akhir
            st.subheader("Hasil Prediksi Data Paling Akhir")
            with st.expander("Klik untuk melihat detail variabel dan hasil prediksi"):
                last_row = sensor_data.iloc[-1]
                st.write("**Variabel Data Paling Akhir:**")
                st.write(last_row[fitur])
                st.write(f"**Prediksi Kebakaran:** {last_row['Prediksi Kebakaran']}")

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

            # Pra-pemrosesan input pengguna
            input_scaled = scaler.transform(input_data)

            # Prediksi untuk input pengguna
            user_prediction = model.predict(input_scaled)
            user_label = convert_to_label(user_prediction[0])

            # Tampilkan hasil prediksi
            if user_label == "High" or user_label == "Very High":
                st.error(f"**Prediksi Risiko Kebakaran: {user_label}!**")
            else:
                st.success(f"**Prediksi Risiko Kebakaran: {user_label}.**")
        else:
            st.error("Data sensor tidak memiliki semua kolom fitur yang diperlukan.")
