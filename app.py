import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image
from streamlit_autorefresh import st_autorefresh

# Load favicon image
im = Image.open("favicon.ico")
st.set_page_config(
    page_title="UHTP Smart Fire Prediction",
    page_icon=im,
)

# CSS untuk latar belakang berwarna cyan, frame merah pada judul, dan frame putih untuk konten
st.markdown("""
    <style>
        /* Latar belakang berwarna cyan */
        .stApp {
            background-color: #E0FFFF; /* Warna latar belakang cyan */
        }
        
        /* Frame putih untuk konten utama */
        .content-frame {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 800px;
            margin: auto;
        }
        
        /* Frame merah untuk judul dan deskripsi */
        .title-frame {
            border: 4px solid red;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        /* Style table untuk hasil data */
        table { width: 100%; }
        thead th { text-align: center; background-color: #f0f0f0; }
        td { text-align: left; }
        th { width: 40%; }
    </style>
""", unsafe_allow_html=True)

# Wrapper untuk judul dan deskripsi dengan frame merah
st.markdown('<div class="title-frame">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.title("UHTP Smart Fire Prediction")

st.markdown("""
    Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan menggunakan pengembangan model Hybrid Machine dan Deep Learning.
    Data diambil dari perangkat IoT secara Realtime [Google Sheets](https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?usp=sharing).
""")
st.markdown('</div>', unsafe_allow_html=True)  # Penutup frame merah untuk judul dan deskripsi

# Wrapper di sekitar konten utama
with st.container():
    st.markdown('<div class="content-frame">', unsafe_allow_html=True)  # Pembuka frame putih

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
    st_autorefresh(interval=3000, limit=None, key="data_refresh")
    st.cache_data.clear()

    # Muat Data
    sensor_data = load_data(data_url)

    # Muat Model dan Scaler
    model = load_model('meta_LR.joblib')
    scaler = load_scaler('scaler.joblib')

    # Tampilkan hasil prediksi data paling akhir sebelum data sensor
    if sensor_data is not None and model is not None and scaler is not None:
        st.subheader("Hasil Prediksi Data Realtime")

        # Mengganti nama kolom sesuai dengan model yang dilatih
        sensor_data = sensor_data.rename(columns={
            'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
            'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
            'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
            'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
            'Kelembapan Tanah': 'Kelembaban Perbukaan Tanah',
            'Waktu': 'Waktu'  # Pastikan ada kolom waktu
        })

        # Fitur yang akan diprediksi
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
                    return "High / Tinggi"
                elif pred == 1:
                    return "Low / Rendah"
                elif pred == 2:
                    return "Moderate / Sedang"
                elif pred == 3:
                    return "Very High / Sangat Tinggi"
                else:
                    return "Unknown"

            sensor_data['Prediksi Kebakaran'] = [convert_to_label(pred) for pred in predictions]

            # Mengambil waktu dari kolom waktu dan format menjadi hari, tanggal, bulan, tahun
            last_row = sensor_data.iloc[-1]
            waktu_prediksi = pd.to_datetime(last_row['Waktu'])
            hari_indonesia = convert_day_to_indonesian(waktu_prediksi.strftime('%A'))
            bulan_indonesia = convert_month_to_indonesian(waktu_prediksi.strftime('%B'))
            tanggal_prediksi = waktu_prediksi.strftime(f'%d {bulan_indonesia} %Y')

            st.write("**Data Sensor Realtime:**")
            sensor_html = pd.DataFrame({
                "Variabel": ["Tavg: Temperatur rata-rata (°C)", "RH_avg: Kelembapan rata-rata (%)", "RR: Curah hujan (mm)", "ff_avg: Kecepatan angin rata-rata (m/s)", "Kelembaban Perbukaan Tanah"],
                "Value": last_row[fitur].values
            }).to_html(index=False)
            st.markdown(sensor_html, unsafe_allow_html=True)

            # Prediksi Kebakaran
            risk = last_row['Prediksi Kebakaran']
            risk_styles = {
                "Low / Rendah": {"color": "white", "background-color": "blue"},
                "Moderate / Sedang": {"color": "white", "background-color": "green"},
                "High / Tinggi": {"color": "black", "background-color": "yellow"},
                "Very High / Sangat Tinggi": {"color": "white", "background-color": "red"}
            }
            risk_style = risk_styles.get(risk, {"color": "black", "background-color": "white"})
            st.markdown(
                f"<p style='color:{risk_style['color']}; background-color:{risk_style['background-color']}; padding: 10px; border-radius: 5px;'>"
                f"Pada hari {hari_indonesia}, tanggal {tanggal_prediksi}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
                f"<span style='font-weight: bold; font-size: 28px; text-decoration: underline;'>{risk}</span></p>", 
                unsafe_allow_html=True
            )

    # Tabel risiko kebakaran
    st.markdown("""
        **Tabel berikut menunjukkan besarnya tingkat resiko kebakaran dan intensitas api jika terjadi kebakaran hutan dan lahan.**
    """)
    st.markdown("""
        <table style="width:100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f0f0f0; text-align: left;">
                    <th style="padding: 8px; border: 1px solid #ddd; width: 15%;">Warna</th>
                    <th style="padding: 8px; border: 1px solid #ddd; width: 20%;">Tingkat Resiko / Intensitas</th>
                    <th style="padding: 8px; border: 1px solid #ddd; width: 65%;">Keterangan</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background-color: blue; color: white;">
                    <td style="padding: 8px; border: 1px solid #ddd;">Blue</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">Low</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran rendah.</td>
                </tr>
                <tr style="background-color: green; color: white;">
                    <td style="padding: 8px; border: 1px solid #ddd;">Green</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">Moderate</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran sedang.</td>
                </tr>
                <tr style="background-color: yellow; color: black;">
                    <td style="padding: 8px; border: 1px solid #ddd;">Yellow</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">High</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran tinggi.</td>
                </tr>
                <tr style="background-color: red; color: white;">
                    <td style="padding: 8px; border: 1px solid #ddd;">Red</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">Very High</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran sangat tinggi.</td>
                </tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

    # Footer dengan logo dan tulisan
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image("kemdikbud.png", width=100)
    with col2:
        st.markdown("<h3 style='text-align: center;'>UHTP Smart Fire Prediction V1</h3>", unsafe_allow_html=True)
    with col3:
        st.image("uhtp.png", width=100)

    st.markdown("<p style='text-align: center;'>Dikembangkan oleh Tim Dosen Universitas Hang Tuah Pekanbaru Tahun 2024</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Penutup frame putih untuk konten utama
