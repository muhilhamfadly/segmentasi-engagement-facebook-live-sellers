import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Segmentasi Engagement Facebook Live",
    page_icon="📊",
    layout="centered",
)

MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "cluster_features.pkl"


# Load model & scaler (di-cache agar tidak reload setiap interaksi)
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
    except FileNotFoundError as e:
        st.error(
            "File model tidak ditemukan. Pastikan `kmeans_model.pkl`, "
            "`scaler.pkl`, dan `cluster_features.pkl` berada di folder yang "
            f"sama dengan app.py.\n\nDetail error: {e}"
        )
        st.stop()
    return model, scaler, features


@st.cache_resource
def get_cluster_label_map(_model, _scaler):
    """
    Menentukan nama/label bisnis untuk tiap indeks cluster secara dinamis
    berdasarkan karakteristik centroid-nya (bukan hardcode index 0/1/2),
    supaya tetap benar walau model dilatih ulang dan urutan cluster berubah.
    """
    centers_scaled = _model.cluster_centers_
    centers_log = _scaler.inverse_transform(centers_scaled)
    centers_original = np.expm1(centers_log)  # kembali ke skala asli

    profile = pd.DataFrame(
        centers_original, columns=["num_reactions", "num_comments", "num_shares"]
    )
    profile["total_engagement"] = profile.sum(axis=1)
    # rasio komentar+share terhadap reaksi -> indikator "diskusi aktif"
    profile["discussion_ratio"] = (profile["num_comments"] + profile["num_shares"]) / (
        profile["num_reactions"] + 1
    )

    # cluster dengan total engagement terendah -> Engagement Rendah
    low_idx = profile["total_engagement"].idxmin()
    remaining = profile.drop(index=low_idx)

    # dari yang tersisa, discussion_ratio tertinggi -> Viral & Diskusi Aktif
    viral_idx = remaining["discussion_ratio"].idxmax()
    # sisanya -> Populer namun Pasif
    passive_idx = remaining.drop(index=viral_idx).index[0]

    label_map = {
        int(low_idx): "Engagement Rendah",
        int(viral_idx): "Viral & Banyak Diperdebatkan",
        int(passive_idx): "Populer namun Pasif",
    }
    color_map = {
        "Engagement Rendah": "#4C78A8",
        "Viral & Banyak Diperdebatkan": "#B26B5C",
        "Populer namun Pasif": "#54C6C6",
    }
    description_map = {
        "Engagement Rendah": (
            "Reaksi, komentar, dan share sama-sama rendah. Konten jenis ini "
            "kurang menarik perhatian audiens, perlu dievaluasi ulang dari "
            "sisi topik, waktu tayang, atau format."
        ),
        "Viral & Banyak Diperdebatkan": (
            "Reaksi, komentar, dan share sama-sama tinggi. Konten ini berhasil "
            "memicu diskusi aktif dan tersebar luas, cocok dijadikan referensi "
            "format live selling interaktif."
        ),
        "Populer namun Pasif": (
            "Reaksi tinggi, tetapi komentar dan share rendah. Konten disukai "
            "secara pasif tanpa memicu diskusi lebih lanjut, cocok untuk "
            "showcase produk singkat."
        ),
    }
    return label_map, color_map, description_map, profile


def predict_cluster(model, scaler, features, reactions, comments, shares):
    X_new = pd.DataFrame([[reactions, comments, shares]], columns=features)
    X_log = np.log1p(X_new)
    X_scaled = scaler.transform(X_log)
    cluster = int(model.predict(X_scaled)[0])
    return cluster


# Load semua artefak
model, scaler, features = load_artifacts()
label_map, color_map, description_map, centroid_profile = get_cluster_label_map(model, scaler)

# Header
st.title("📊 Segmentasi Engagement Facebook Live Sellers")
st.markdown(
    "Masukkan metrik engagement sebuah postingan untuk mengetahui segmennya, "
    "berdasarkan model **K-Means (K=3)** yang dilatih pada dataset "
    "[Facebook Live Sellers in Thailand](https://www.kaggle.com/datasets/ashishg21/facebook-live-sellers-in-thailand-uci-ml-repo/data)."
)

st.subheader("Masukkan Metrik Engagement Postingan")

col1, col2, col3 = st.columns(3)
with col1:
    reactions = st.number_input("Jumlah Reaksi (num_reactions)", min_value=0, value=100, step=1)
with col2:
    comments = st.number_input("Jumlah Komentar (num_comments)", min_value=0, value=10, step=1)
with col3:
    shares = st.number_input("Jumlah Share (num_shares)", min_value=0, value=2, step=1)

if st.button("🔮 Prediksi Segmen", type="primary"):
    cluster = predict_cluster(model, scaler, features, reactions, comments, shares)
    label = label_map[cluster]
    color = color_map[label]
    desc = description_map[label]

    st.markdown("### Hasil Prediksi")
    st.markdown(
        f"""
        <div style="padding:1.2rem;border-radius:0.6rem;background-color:{color}22;
                    border-left:6px solid {color};">
            <h4 style="margin:0;color:{color};">Segmen: {label}</h4>
            <p style="margin-top:0.5rem;">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Posisikan input di antara profil centroid untuk konteks
    st.markdown("#### Perbandingan dengan Profil Rata-rata Tiap Segmen")
    compare_df = centroid_profile[["num_reactions", "num_comments", "num_shares"]].copy()
    compare_df.index = [label_map[i] for i in compare_df.index]
    compare_df.loc["Postingan Anda"] = [reactions, comments, shares]
    st.dataframe(compare_df.round(1), use_container_width=True)
