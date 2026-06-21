import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Analisis Matriks Wajah & Kompresi PCA", layout="wide")

st.title("🧮 Analisis Matriks Wajah & Kompresi Gambar PCA")
st.caption("Proyek Aljabar Linear — Verifikasi wajah (DeepFace) & Kompresi gambar (PCA)")

tab1, tab2 = st.tabs(["🔍 Verifikasi Wajah", "🗜️ Kompresi PCA"])


def load_image_cv2(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ───────────────────────────────────────────────
# TAB 1: VERIFIKASI WAJAH (Aljabar.py)
# ───────────────────────────────────────────────
with tab1:
    st.header("Verifikasi Wajah")
    st.write("Upload dua foto wajah untuk dibandingkan kemiripannya.")

    col1, col2 = st.columns(2)
    with col1:
        foto1 = st.file_uploader("Foto 1", type=["jpg", "jpeg", "png"], key="foto1")
        if foto1:
            st.image(foto1, caption="Foto 1", use_container_width=True)
    with col2:
        foto2 = st.file_uploader("Foto 2", type=["jpg", "jpeg", "png"], key="foto2")
        if foto2:
            st.image(foto2, caption="Foto 2", use_container_width=True)

    if foto1 and foto2:
        if st.button("Analisis Sekarang", type="primary"):
            from deepface import DeepFace

            path1 = path2 = None
            with st.spinner("Menganalisis matriks wajah... mohon tunggu."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t1:
                        t1.write(foto1.getvalue())
                        path1 = t1.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t2:
                        t2.write(foto2.getvalue())
                        path2 = t2.name

                    hasil = DeepFace.verify(img1_path=path1, img2_path=path2, enforce_detection=True)
                    is_match = hasil["verified"]
                    distance = hasil["distance"]
                    threshold = hasil.get("threshold", 0.40)

                    if is_match:
                        persentase = 100.0 - ((distance / threshold) * 50.0)
                    else:
                        persentase = (threshold / distance) * 50.0
                    persentase = min(100.0, persentase)

                    if is_match:
                        st.success(f"✅ COCOK — Kemiripan: {persentase:.2f}% (Jarak Matriks: {distance:.4f})")
                    else:
                        st.error(f"❌ TIDAK COCOK — Kemiripan: {persentase:.2f}% (Jarak Matriks: {distance:.4f})")

                    img1_bgr = cv2.imread(path1)
                    img2_bgr = cv2.imread(path2)
                    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
                    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
                    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
                    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    status = "COCOK" if is_match else "TIDAK COCOK"
                    fig.suptitle(
                        f"Hasil Analisis: {status}\nKemiripan: {persentase:.2f}% (Jarak: {distance:.4f})",
                        fontsize=14, fontweight="bold"
                    )
                    axes[0, 0].imshow(img1_gray, cmap="gray")
                    axes[0, 0].set_title("Foto 1: Grayscale (Matriks 2D)")
                    axes[0, 1].imshow(img2_gray, cmap="gray")
                    axes[0, 1].set_title("Foto 2: Grayscale (Matriks 2D)")
                    axes[1, 0].imshow(img1_rgb)
                    axes[1, 0].set_title("Foto 1: Berwarna (Matriks RGB)")
                    axes[1, 1].imshow(img2_rgb)
                    axes[1, 1].set_title("Foto 2: Berwarna (Matriks RGB)")
                    for ax in axes.flat:
                        ax.axis("off")
                    plt.tight_layout()
                    st.pyplot(fig)

                except ValueError:
                    st.error("⚠️ Wajah tidak terdeteksi. Pastikan foto jelas dan tidak tertutup.")
                except Exception as e:
                    st.error(f"⚠️ Terjadi kesalahan: {e}")
                finally:
                    for p in (path1, path2):
                        if p and os.path.exists(p):
                            os.unlink(p)
    else:
        st.info("Upload kedua foto terlebih dahulu untuk mulai analisis.")


# ───────────────────────────────────────────────
# TAB 2: KOMPRESI PCA (detector.py)
# ───────────────────────────────────────────────
with tab2:
    st.header("Kompresi Gambar dengan PCA")
    st.write("Upload satu gambar untuk dikompres menggunakan PCA pada matriks grayscale.")

    foto = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"], key="foto_pca")
    k = st.slider("Jumlah komponen utama (k)", min_value=5, max_value=300, value=50, step=5,
                   help="Semakin besar k, gambar makin jelas tapi data yang disimpan makin banyak.")

    if foto:
        img_bgr = load_image_cv2(foto)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if st.button("Jalankan Kompresi", type="primary"):
            with st.spinner(f"Memproses matriks grayscale dengan k = {k}..."):
                tinggi, lebar = img_gray.shape

                rata_rata = np.mean(img_gray, axis=0)
                matriks_pusat = img_gray - rata_rata

                kovarians = np.cov(matriks_pusat, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(kovarians)

                urutan = np.argsort(eigenvalues)[::-1]
                eigenvectors_urut = eigenvectors[:, urutan]
                komponen_utama = eigenvectors_urut[:, :k]

                matriks_terkompresi = np.dot(matriks_pusat, komponen_utama)
                matriks_rekonstruksi = np.dot(matriks_terkompresi, komponen_utama.T) + rata_rata

                data_asli = tinggi * lebar
                data_pca = (lebar * k) + (tinggi * k) + lebar
                persen_hemat = 100 - ((data_pca / data_asli) * 100)

                img_hasil = np.clip(matriks_rekonstruksi, 0, 255).astype(np.uint8)

                c1, c2, c3 = st.columns(3)
                c1.metric("Data Matriks Asli", f"{data_asli:,}")
                c2.metric("Data Matriks PCA", f"{data_pca:,}")
                c3.metric("Penghematan", f"{persen_hemat:.2f}%")

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Kompresi Grayscale PCA (k={k})", fontsize=14, fontweight="bold")
                axes[0].imshow(img_gray, cmap="gray")
                axes[0].set_title("Grayscale Asli")
                axes[0].axis("off")
                axes[1].imshow(img_hasil, cmap="gray")
                axes[1].set_title(f"Hasil PCA k={k}")
                axes[1].axis("off")
                plt.tight_layout()
                st.pyplot(fig)

                ok, buffer = cv2.imencode(".png", img_hasil)
                if ok:
                    st.download_button(
                        "⬇️ Download hasil kompresi",
                        data=buffer.tobytes(),
                        file_name=f"hasil_pca_k{k}.png",
                        mime="image/png"
                    )
    else:
        st.info("Upload gambar terlebih dahulu untuk mulai kompresi.")
