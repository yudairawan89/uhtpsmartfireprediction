# Import Streamlit dan pustaka yang diperlukan
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import io

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
        st.write("Data berhasil dimuat.")  # Pesan debugging
        return data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.write("Model berhasil dimuat.")  # Pesan debugging
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Fungsi untuk memuat scaler yang sudah dilatih
@st.cache_resource
def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        st.write("Scaler berhasil dimuat.")  # Pesan debugging
        return scaler
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat scaler: {e}")
        return None

# URL Data Google Sheets (format CSV)
data_url = 'https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv'

# Tombol untuk refresh data
if st.button('Refresh Data'):
    st.cache_data.clear()  # Hapus cache agar data terbaru dimuat

# Muat Data
sensor_data = load_data(data_url)

# Muat Model dan Scaler
model = load_model('meta_LR.joblib')
scaler = load_scaler('scaler.joblib')

# Periksa apakah data sensor berhasil dimuat
if sensor_data is not None:
    st.subheader("Data Sensor")
    st.write(f"Kolom yang tersedia: {sensor_data.columns.tolist()}")  # Menampilkan kolom yang ada di data untuk debugging
    st.dataframe(sensor_data)  # Tampilkan data sensor

    # Mengganti nama kolom sesuai dengan model yang dilatih
    sensor_data = sensor_data.rename(columns={
        'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
        'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Perbukaan Tanah',
        'Waktu': 'Waktu'  # Pastikan ada kolom waktu
    })

# Periksa apakah model dan scaler berhasil dimuat
if model is not None and scaler is not None:
    st.write("Model dan scaler berhasil dimuat. Siap untuk prediksi.")

    # Fitur yang akan diprediksi
    fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)',
             'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Perbukaan Tanah']

    # Periksa apakah semua kolom fitur tersedia
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

        # Menampilkan hasil prediksi data terakhir
        last_row = sensor_data.iloc[-1]
        st.subheader("Hasil Prediksi Data Paling Akhir")
        st.write(last_row)

    else:
        st.error("Data sensor tidak memiliki semua kolom fitur yang diperlukan.")

    # Fitur download hasil prediksi sebagai CSV
    csv = sensor_data.to_csv(index=False)
    st.download_button(
        label="Download Hasil Prediksi sebagai CSV",
        data=csv,
        file_name='hasil_prediksi_kebakaran.csv',
        mime='text/csv'
    )

# Jika data atau model/scaler tidak dimuat dengan benar
else:
    st.error("Data, model, atau scaler tidak berhasil dimuat. Tidak bisa melakukan prediksi.")
