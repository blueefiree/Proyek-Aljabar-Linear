import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

def analisis_visual_wajah(path_foto1, path_foto2):
    print("Menganalisis matriks wajah... Mohon tunggu.")
    
    try:
        # 1. PROSES DEEPFACE (Analisis AI)
        hasil = DeepFace.verify(
            img1_path=path_foto1, 
            img2_path=path_foto2,
            enforce_detection=True
        )
        
        is_match = hasil['verified']
        distance = hasil['distance']
        
        # --- BAGIAN KODE BARU ---
        # Mengambil threshold bawaan model (biasanya 0.40 untuk VGG-Face)
        threshold = hasil.get('threshold', 0.40)
        
        # Perhitungan persentase berbasis threshold
        if is_match:
            # Jika cocok, persentase dihitung dari 100% turun hingga batas 50% di titik threshold
            persentase_kemiripan = 100.0 - ((distance / threshold) * 50.0)
        else:
            # Jika tidak cocok, persentase berada di bawah 50%
            persentase_kemiripan = (threshold / distance) * 50.0
            
        # Memastikan nilai tidak lebih dari 100%
        persentase_kemiripan = min(100.0, persentase_kemiripan)
        # --- AKHIR BAGIAN KODE BARU ---

        # 2. PROSES PEMBACAAN MATRIKS GAMBAR (OpenCV)
        # Membaca gambar asli (OpenCV membacanya dalam format BGR - Blue, Green, Red)
        img1_bgr = cv2.imread(path_foto1)
        img2_bgr = cv2.imread(path_foto2)
        
        # Konversi ke Grayscale (Matriks 2D tunggal)
        img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
        
        # Konversi ke RGB (Matriks 3D untuk ditampilkan kembali dengan warna asli)
        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

        # 3. PROSES VISUALISASI (Matplotlib)
        # Membuat jendela pop-up dengan ukuran 10x8 dan layout 2x2
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Mengatur Judul Utama (Menampilkan Persentase)
        status = "✅ COCOK" if is_match else "❌ TIDAK COCOK"
        fig.suptitle(f"Hasil Analisis: {status}\nKemiripan: {persentase_kemiripan:.2f}% (Jarak Matriks: {distance:.4f})", 
                     fontsize=14, fontweight='bold')

        # --- Baris Atas: Grayscale (Proses Komputer) ---
        axes[0, 0].imshow(img1_gray, cmap='gray')
        axes[0, 0].set_title("Foto 1: Grayscale (Matriks 2D)")
        
        axes[0, 1].imshow(img2_gray, cmap='gray')
        axes[0, 1].set_title("Foto 2: Grayscale (Matriks 2D)")

        # --- Baris Bawah: Berwarna (Dikembalikan ke Asli) ---
        axes[1, 0].imshow(img1_rgb)
        axes[1, 0].set_title("Foto 1: Berwarna (Matriks RGB)")
        
        axes[1, 1].imshow(img2_rgb)
        axes[1, 1].set_title("Foto 2: Berwarna (Matriks RGB)")

        # Menghilangkan garis sumbu X dan Y agar gambar terlihat bersih
        for ax in axes.flat:
            ax.axis('off')

        # Menampilkan pop-up ke layar
        plt.tight_layout()
        plt.show()

    except ValueError:
        print("\n⚠️ ERROR: Wajah tidak terdeteksi. Pastikan foto jelas dan tidak tertutup.")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")

# --- Eksekusi Program ---
if __name__ == "__main__":
    # Ganti dengan nama file foto kamu yang ada di folder yang sama
    file_lama = "foto_lama.jpg"
    file_baru = "foto_sekarang.jpg"
    
    analisis_visual_wajah(file_lama, file_baru)