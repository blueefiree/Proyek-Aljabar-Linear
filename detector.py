# import cv2
# import matplotlib.pyplot as plt
# from deepface import DeepFace

# def analisis_visual_wajah(path_foto1, path_foto2):
#     print("Menganalisis matriks wajah... Mohon tunggu.")
    
#     try:
#         # 1. PROSES DEEPFACE (Analisis AI)
#         hasil = DeepFace.verify(
#             img1_path=path_foto1, 
#             img2_path=path_foto2,
#             enforce_detection=True
#         )
        
#         is_match = hasil['verified']
#         distance = hasil['distance']
        
#         # --- BAGIAN KODE BARU ---
#         # Mengambil threshold bawaan model (biasanya 0.40 untuk VGG-Face)
#         threshold = hasil.get('threshold', 0.40)
        
#         # Perhitungan persentase berbasis threshold
#         if is_match:
#             # Jika cocok, persentase dihitung dari 100% turun hingga batas 50% di titik threshold
#             persentase_kemiripan = 100.0 - ((distance / threshold) * 50.0)
#         else:
#             # Jika tidak cocok, persentase berada di bawah 50%
#             persentase_kemiripan = (threshold / distance) * 50.0
            
#         # Memastikan nilai tidak lebih dari 100%
#         persentase_kemiripan = min(100.0, persentase_kemiripan)
#         # --- AKHIR BAGIAN KODE BARU ---

#         # 2. PROSES PEMBACAAN MATRIKS GAMBAR (OpenCV)
#         # Membaca gambar asli (OpenCV membacanya dalam format BGR - Blue, Green, Red)
#         img1_bgr = cv2.imread(path_foto1)
#         img2_bgr = cv2.imread(path_foto2)
        
#         # Konversi ke Grayscale (Matriks 2D tunggal)
#         img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
#         img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
        
#         # Konversi ke RGB (Matriks 3D untuk ditampilkan kembali dengan warna asli)
#         img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
#         img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

#         # 3. PROSES VISUALISASI (Matplotlib)
#         # Membuat jendela pop-up dengan ukuran 10x8 dan layout 2x2
#         fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
#         # Mengatur Judul Utama (Menampilkan Persentase)
#         status = "✅ COCOK" if is_match else "❌ TIDAK COCOK"
#         fig.suptitle(f"Hasil Analisis: {status}\nKemiripan: {persentase_kemiripan:.2f}% (Jarak Matriks: {distance:.4f})", 
#                      fontsize=14, fontweight='bold')

#         # --- Baris Atas: Grayscale (Proses Komputer) ---
#         axes[0, 0].imshow(img1_gray, cmap='gray')
#         axes[0, 0].set_title("Foto 1: Grayscale (Matriks 2D)")
        
#         axes[0, 1].imshow(img2_gray, cmap='gray')
#         axes[0, 1].set_title("Foto 2: Grayscale (Matriks 2D)")

#         # --- Baris Bawah: Berwarna (Dikembalikan ke Asli) ---
#         axes[1, 0].imshow(img1_rgb)
#         axes[1, 0].set_title("Foto 1: Berwarna (Matriks RGB)")
        
#         axes[1, 1].imshow(img2_rgb)
#         axes[1, 1].set_title("Foto 2: Berwarna (Matriks RGB)")

#         # Menghilangkan garis sumbu X dan Y agar gambar terlihat bersih
#         for ax in axes.flat:
#             ax.axis('off')

#         # Menampilkan pop-up ke layar
#         plt.tight_layout()
#         plt.show()

#     except ValueError:
#         print("\n⚠️ ERROR: Wajah tidak terdeteksi. Pastikan foto jelas dan tidak tertutup.")
#     except Exception as e:
#         print(f"\n⚠️ Terjadi kesalahan: {e}")

# # --- Eksekusi Program ---
# if __name__ == "__main__":
#     # Ganti dengan nama file foto kamu yang ada di folder yang sama
#     file_lama = "foto_lama.jpg"
#     file_baru = "foto_sekarang.jpg"
    
#     analisis_visual_wajah(file_lama, file_baru)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def kompresi_pca_grayscale(matriks_gray, k):
    """
    Mengompres 1 matriks (grayscale) menggunakan PCA
    """
    tinggi, lebar = matriks_gray.shape
    
    # 1. Standardisasi (Pemusatan data ke rata-rata)
    rata_rata = np.mean(matriks_gray, axis=0) 
    matriks_pusat = matriks_gray - rata_rata
    
    # 2. Proses PCA (Kovarians & Eigenvalue)
    kovarians = np.cov(matriks_pusat, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(kovarians)
    
    # 3. Mengurutkan dan mengambil k komponen terbaik
    urutan = np.argsort(eigenvalues)[::-1]
    eigenvectors_urut = eigenvectors[:, urutan]
    komponen_utama = eigenvectors_urut[:, :k]
    
    # 4. Kompresi & Rekonstruksi Matriks
    matriks_terkompresi = np.dot(matriks_pusat, komponen_utama)
    matriks_rekonstruksi = np.dot(matriks_terkompresi, komponen_utama.T) + rata_rata
    
    # Menghitung titik data secara matematis
    data_asli = tinggi * lebar 
    data_pca = (lebar * k) + (tinggi * k) + lebar
    
    return matriks_rekonstruksi, data_asli, data_pca

def jalankan_kompresi_gray(image_path, output_path, k_components=100):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} tidak ditemukan!")
        return

    # Membaca gambar dan LANGSUNG mengubahnya ke Grayscale
    img_bgr = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    print(f"Memproses 1 matriks Grayscale dengan k = {k_components}...")
    
    # Menjalankan PCA pada satu matriks saja
    img_rekonstruksi, data_asli, data_pca = kompresi_pca_grayscale(img_gray, k_components)
    
    # Memastikan nilai piksel valid (0-255)
    img_hasil = np.clip(img_rekonstruksi, 0, 255).astype(np.uint8)
    
    # Menyimpan file hasil (kualitas JPEG diatur ke 80 agar ukuran disk efisien)
    cv2.imwrite(output_path, img_hasil, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    
    # --- KALKULASI HASIL ---
    persen_hemat_matriks = 100 - ((data_pca / data_asli) * 100)
    
    ukuran_disk_asli = os.path.getsize(image_path) / 1024
    ukuran_disk_hasil = os.path.getsize(output_path) / 1024

    print("\n" + "="*45)
    print("HASIL KOMPRESI GRAYSCALE PCA")
    print("="*45)
    print(f"Data Matriks Asli  : {data_asli:,} parameter")
    print(f"Data Matriks PCA   : {data_pca:,} parameter")
    print(f"Penghematan Matriks: {persen_hemat_matriks:.2f}%")
    print("-" * 45)
    print(f"Ukuran Disk Asli   : {ukuran_disk_asli:.2f} KB (Berwarna)")
    print(f"Ukuran Disk Hasil  : {ukuran_disk_hasil:.2f} KB (Grayscale PCA)")
    print("="*45)

    # --- VISUALISASI ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Kompresi Grayscale PCA (k={k_components})\nPenghematan Data: {persen_hemat_matriks:.2f}%", 
                 fontsize=14, fontweight='bold')
    
    # Perhatikan penambahan cmap='gray' agar Matplotlib tidak mewarnainya jadi kuning/ungu
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title(f"Grayscale Asli\nDisk: {ukuran_disk_asli:.2f} KB")
    axes[0].axis('off')
    
    axes[1].imshow(img_hasil, cmap='gray')
    axes[1].set_title(f"Grayscale Hasil PCA\nDisk: {ukuran_disk_hasil:.2f} KB")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Kamu bisa mengatur nilai k_components
    # Semakin besar angkanya, wajah semakin jelas tapi file lebih besar
    jalankan_kompresi_gray("foto_saya.png", "foto_grayscale_pca.png", k_components=50)