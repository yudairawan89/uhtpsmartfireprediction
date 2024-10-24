# Tampilkan hasil prediksi data paling akhir sebelum data sensor
if sensor_data is not None and model is not None and scaler is not None:
    st.subheader("Hasil Prediksi Data Paling Realtime")

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

        # Mengambil waktu dari kolom waktu dan format menjadi hari, tanggal, bulan, tahun
        last_row = sensor_data.iloc[-1]
        waktu_prediksi = pd.to_datetime(last_row['Waktu'])
        hari_indonesia = convert_day_to_indonesian(waktu_prediksi.strftime('%A'))
        bulan_indonesia = convert_month_to_indonesian(waktu_prediksi.strftime('%B'))
        tanggal_prediksi = waktu_prediksi.strftime(f'%d {bulan_indonesia} %Y')

        # Menambahkan kolom Variabel dan Value sesuai permintaan user
        st.write("**Data Sensor Realtime:**")
        st.table(pd.DataFrame({
            "Variabel": ["Tavg: Temperatur rata-rata (°C)", "RH_avg: Kelembapan rata-rata (%)", "RR: Curah hujan (mm)", "ff_avg: Kecepatan angin rata-rata (m/s)", "Kelembaban Perbukaan Tanah"],
            "Value": last_row[fitur].values
        }).set_index("Variabel"))  # Hide the row numbers by setting the "Variabel" column as the index
