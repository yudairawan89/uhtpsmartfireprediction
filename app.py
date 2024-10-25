# Import pustaka yang diperlukan
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import io
import time

# Fungsi untuk mengonversi hari ke bahasa Indonesia
def convert_day_to_indonesian(day_name):
    days_translation = {
        'Monday': 'Senin',
        'Tuesday': 'Selasa',
        'Wednesday': 'Rabu',
        'Thursday': 'Kamis',
        'Friday': 'Jumat',
        'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }
    return days_translation.get(day_name, day_name)

# Fungsi untuk mengonversi bulan ke bahasa Indonesia
def convert_month_to_indonesian(month_name):
    months_translation = {
        'January': 'Januari',
        'February': 'Februari',
        'March': 'Maret',
        'April': 'April',
        'May': 'Mei',
        'June': 'Juni',
        'July': 'Juli',
        'August': 'Agustus',
        'September': 'September',
        'October': 'Oktober',
        'November': 'November',
        'December': 'Desember'
    }
    return months_translation.get(month_name, month_name)

# Menambahkan logo di sebelah kiri tulisan "UHTP Smart Fire Prediction"
col1, col2 = st.columns([1, 6])  # Membuat layout kolom untuk logo dan judul
with col1:
    st.image("logo.png", width=100)  # Menambahkan logo dari folder yang sama dengan aplikasi
with col2:
    # Judul Aplikasi
    st.title("UHTP Smart Fire Prediction")

# Deskripsi Aplikasi
st.markdown("""
Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan menggunakan pengembangan model Hybrid Machine dan Deep Learning.
Data diambil dari perangkat IoT secara Realtime [Google Sheets](https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?usp=sharing).
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

# Buat loop untuk auto-refresh setiap 3 detik
while True:
    sensor_data = load_data(data_url)

    # Muat Model dan Scaler
    model = load_model('meta_LR.joblib')
    scaler = load_scaler('scaler.joblib')

    if sensor_data is not None and model is not None and scaler is not None:
        st.subheader("Hasil Prediksi Data Realtime")

        # Mengganti nama kolom sesuai dengan model yang dilatih
        sensor_data = sensor_data.rename(columns={
            'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
            'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
            'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
            'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
            'Kelembapan Tanah': 'Kelembaban Perbukaan Tanah',
            'Waktu': 'Waktu'
        })

        # Prediksi
        fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)',
                 'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Perbukaan Tanah']

        if all(col in sensor_data.columns for col in fitur):
            fitur_data = sensor_data[fitur]

            for col in fitur_data.columns:
                fitur_data[col] = fitur_data[col].astype(str).str.replace(',', '.').astype(float)

            fitur_data.fillna(0, inplace=True)

            fitur_scaled = scaler.transform(fitur_data)
            fitur_scaled_df = pd.DataFrame(fitur_scaled, columns=fitur)

            predictions = model.predict(fitur_scaled_df)

            sensor_data['Prediksi Kebakaran'] = [convert_to_label(pred) for pred in predictions]

            last_row = sensor_data.iloc[-1]
            waktu_prediksi = pd.to_datetime(last_row['Waktu'])
            hari_indonesia = convert_day_to_indonesian(waktu_prediksi.strftime('%A'))
            bulan_indonesia = convert_month_to_indonesian(waktu_prediksi.strftime('%B'))
            tanggal_prediksi = waktu_prediksi.strftime(f'%d {bulan_indonesia} %Y')

            st.markdown(f"<p>Hasil prediksi untuk {hari_indonesia}, {tanggal_prediksi}</p>", unsafe_allow_html=True)
            
            st.write(sensor_data)

            # Tunggu 3 detik sebelum refresh berikutnya
            time.sleep(3)
            st.experimental_rerun()  # Untuk mererun script

