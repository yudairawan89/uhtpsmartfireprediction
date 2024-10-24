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

# Tombol untuk refresh data
if st.button('Refresh Data'):
    st.cache_data.clear()  # Hapus cache agar data terbaru dimuat

# Muat Data
sensor_data = load_data(data_url)

# Tampilkan hasil prediksi data paling akhir setelah tombol refresh data
if sensor_data is not None:
    st.subheader("Hasil Prediksi Data Paling Akhir")
    with st.expander("Klik untuk melihat detail variabel dan hasil prediksi"):
        last_row = sensor_data.iloc[-1]

        # Mengambil waktu dari kolom waktu dan format menjadi hari, tanggal, bulan, tahun
        waktu_prediksi = pd.to_datetime(last_row['Waktu'])
        hari_indonesia = convert_day_to_indonesian(waktu_prediksi.strftime('%A'))
        bulan_indonesia = convert_month_to_indonesian(waktu_prediksi.strftime('%B'))
        tanggal_prediksi = waktu_prediksi.strftime(f'%d {bulan_indonesia} %Y')

        st.write("**Variabel Data Paling Akhir:**")
        fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)',
                 'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Perbukaan Tanah']
        st.write(last_row[fitur])

        # Prediksi Kebakaran berdasarkan risiko
        risk = last_row['Prediksi Kebakaran']
        risk_styles = {
            "Low": {"color": "white", "background-color": "blue"},
            "Moderate": {"color": "white", "background-color": "green"},
            "High": {"color": "black", "background-color": "yellow"},
            "Very High": {"color": "white", "background-color": "red"}
        }

        risk_style = risk_styles.get(risk, {"color": "black", "background-color": "white"})

        # Menampilkan prediksi kebakaran dengan indikator risiko lebih besar, tebal, dan garis bawah
        st.markdown(
            f"<p style='color:{risk_style['color']}; background-color:{risk_style['background-color']}; padding: 10px; border-radius: 5px;'>"
            f"Pada hari {hari_indonesia}, tanggal {tanggal_prediksi}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
            f"<span style='font-weight: bold; font-size: 28px; text-decoration: underline;'>{risk}</span></p>", 
            unsafe_allow_html=True
        )

    # Lanjutkan menampilkan data sensor dan hasil prediksi setelah menampilkan bagian ini
    st.subheader("Data Sensor")
    st.dataframe(sensor_data)

    # Mengganti nama kolom sesuai dengan model yang dilatih
    sensor_data = sensor_data.rename(columns={
        'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
        'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Perbukaan Tanah',
        'Waktu': 'Waktu'  # Pastikan ada kolom waktu
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
        else:
            st.error("Data sensor tidak memiliki semua kolom fitur yang diperlukan.")

# Footer dengan logo dan tulisan
st.markdown("---")  # Garis pembatas untuk memisahkan footer
col1, col2, col3 = st.columns([1, 3, 1])  # Layout kolom untuk gambar logo dan teks
with col1:
    st.image("kemdikbud.png", width=100)  # Menampilkan logo Kemdikbud
with col2:
    st.markdown("<h3 style='text-align: center;'>UHTP Smart Fire Prediction - 2024</h3>", unsafe_allow_html=True)
with col3:
    st.image("uhtp.png", width=100)  # Menampilkan logo UHTP

st.markdown("<p style='text-align: center;'>Dikembangkan oleh Universitas Hang Tuah Pekanbaru</p>", unsafe_allow_html=True)
