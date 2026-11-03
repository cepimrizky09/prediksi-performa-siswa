
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Prediksi Performa Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
<style>

/* Background utama */
.stApp {
    background-color: #f4f7fb;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #123c69, #1e88e5);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.header-box h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 5px;
}

.header-box p {
    color: #e3f2fd;
    font-size: 16px;
}

/* Card informasi */
.info-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #1e88e5;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}

/* Section */
.section-title {
    color: #1E293B;
    font-size: 21px;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Tombol */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1e88e5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0d47a1, #1565c0);
    color: white;
    border: none;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #eaf1f8;
}

/* Footer */
.footer {
    text-align: center;
    color: #7b8794;
    font-size: 13px;
    margin-top: 40px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. HEADER APLIKASI
# ==========================================
st.markdown("""
<div class="header-box">
    <h1>🎓 Aplikasi Prediksi Performa Siswa</h1>
    <p>
        Sistem prediksi performa akademik siswa
        menggunakan algoritma Machine Learning Decision Tree.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR INFORMASI
# ==========================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=100
    )

    st.title("Menu Aplikasi")

    st.markdown("---")

    st.info("""
    **Panduan Penggunaan**

    1. Siapkan dataset siswa dalam format CSV.
    2. Upload dataset melalui menu utama.
    3. Periksa preview data.
    4. Klik tombol Jalankan Prediksi.
    5. Lihat hasil prediksi siswa.
    """)

    st.markdown("---")

    st.caption("Model Machine Learning")
    st.success("Decision Tree Classifier")

    st.markdown("---")
    st.caption("Versi Aplikasi 1.0")


# ==========================================
# 5. MEMUAT MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("model_decision_tree.pkl")

try:
    model = load_model()
    model_status = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    model_status = False


# ==========================================
# 6. UPLOAD DATASET
# ==========================================
st.markdown(
    '<div class="section-title">📁 Upload Dataset Siswa</div>',
    unsafe_allow_html=True
)

st.write(
    "Silakan unggah file dataset siswa dengan format CSV "
    "untuk memulai proses prediksi."
)

uploaded_file = st.file_uploader(
    "Pilih file dataset CSV",
    type=["csv"],
    help="Pastikan file memiliki struktur kolom yang sesuai dengan data pelatihan model."
)


# ==========================================
# 7. MEMBACA DATASET
# ==========================================
if uploaded_file is not None:

    try:
        df_user = pd.read_csv(uploaded_file)

        if df_user.empty:
            st.warning("Dataset yang diunggah tidak memiliki data.")
            st.stop()

        st.success("Dataset berhasil diunggah!")

        # Informasi dataset
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Jumlah Siswa",
                value=f"{len(df_user):,}"
            )

        with col2:
            st.metric(
                label="Jumlah Kolom",
                value=len(df_user.columns)
            )

        with col3:
            st.metric(
                label="Status Dataset",
                value="Berhasil"
            )

        # Preview dataset
        st.markdown(
            '<div class="section-title">📊 Preview Dataset</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df_user.head(10),
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Lihat informasi struktur dataset"):
            st.write("Nama Kolom:")
            st.write(list(df_user.columns))

            st.write("Informasi tipe data:")
            st.dataframe(
                df_user.dtypes.astype(str).rename("Tipe Data"),
                use_container_width=True
            )

        st.markdown("---")

        # ==========================================
        # 8. TOMBOL PREDIKSI
        # ==========================================
        st.markdown(
            '<div class="section-title">🤖 Proses Prediksi</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🚀 JALANKAN PREDIKSI",
            disabled=not model_status
        ):

            try:
                with st.spinner(
                    "Sedang memproses data dan melakukan prediksi..."
                ):

                    # Salin dataset
                    df_model_input = df_user.copy()

                    # Hapus kolom target GRADE
                    if "GRADE" in df_model_input.columns:
                        df_model_input = df_model_input.drop(
                            columns=["GRADE"]
                        )

                    # Encoding data kategorikal
                    df_encoded = pd.get_dummies(df_model_input)

                    # Menyesuaikan fitur dengan model
                    df_encoded = df_encoded.reindex(
                        columns=model.feature_names_in_,
                        fill_value=0
                    )

                    # Melakukan prediksi
                    predictions = model.predict(df_encoded)

                    # Menambahkan hasil prediksi
                    df_result = df_user.copy()
                    df_result["Hasil_Prediksi"] = predictions

                st.balloons()

                st.success(
                    "Prediksi berhasil dilakukan!"
                )

                # ==========================================
                # 9. MENAMPILKAN HASIL
                # ==========================================

                st.markdown(
                    '<div class="section-title">🎉 Hasil Prediksi Performa Siswa</div>',
                    unsafe_allow_html=True
                )

                # Statistik hasil prediksi
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Total Siswa Diprediksi",
                        len(df_result)
                    )

                with col2:
                    st.metric(
                        "Kategori Prediksi",
                        df_result["Hasil_Prediksi"].nunique()
                    )

                # Tabel hasil
                st.dataframe(
                    df_result,
                    use_container_width=True,
                    hide_index=True
                )

                # Distribusi hasil prediksi
                st.markdown(
                    '<div class="section-title">📈 Distribusi Hasil Prediksi</div>',
                    unsafe_allow_html=True
                )

                prediction_counts = (
                    df_result["Hasil_Prediksi"]
                    .value_counts()
                    .rename_axis("Kategori")
                    .reset_index(name="Jumlah Siswa")
                )

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.bar_chart(
                        prediction_counts.set_index("Kategori")
                    )

                with col2:
                    st.dataframe(
                        prediction_counts,
                        use_container_width=True,
                        hide_index=True
                    )

                # ==========================================
                # 10. DOWNLOAD HASIL
                # ==========================================

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">📥 Download Hasil Prediksi</div>',
                    unsafe_allow_html=True
                )

                csv_result = df_result.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    label="⬇️ Download Hasil Prediksi (CSV)",
                    data=csv_result,
                    file_name="hasil_prediksi_siswa.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(
                    f"Terjadi kesalahan saat memproses data: {e}"
                )

    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")

else:
    st.info(
        "👆 Silakan upload dataset CSV terlebih dahulu "
        "untuk memulai proses prediksi."
    )


# ==========================================
# 11. FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    🎓 Aplikasi Prediksi Performa Siswa<br>
    Powered by Python, Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
