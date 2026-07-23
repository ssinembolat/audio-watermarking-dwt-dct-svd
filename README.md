# 🎵 Ses Filigranlama Sistemi (Audio Watermarking System)
### DWT + DCT + SVD + QIM + Hamming (7,4) Hata Düzeltme Kodu

Bu proje; Dijital Hak Yönetimi (DRM), telif hakkı koruması ve veri gizleme amacıyla geliştirilmiş **hibrit bir ses filigranlama (audio watermarking) uygulamasıdır**. 

Ses sinyallerinde yüksek **duyulmazlık (imperceptibility)** ve çeşitli sinyal işleme saldırılarına karşı güçlü **dayanıklılık (robustness)** sağlamak için frekans alanı dönüşümleri ile hata düzeltme algoritmalarını bir araya getirir.

---

## 🛠️ Kullanılan Teknolojiler ve Yöntemler

* **DWT (Ayrık Dalgacık Dönüşümü):** Ses sinyali alt frekans bantlarına ayrılarak yüksek enerjili katsayılar elde edilir.
* **2D-DCT (İki Boyutlu Ayrık Kosinüs Dönüşümü):** DWT katsayıları matris formatına getirilerek frekans bileşenlerine dönüştürülür.
* **SVD (Tekil Değer Ayrışımı):** DCT matrisinin tekil değerleri ($S$ matrisi) çıkarılır. Filigran doğrudan bu kararlı değerlere gömülür.
* **QIM (Miktar İndisli Modülasyon):** Veri gömme işlemi adaptif $\alpha$ parametresi ile QIM mantığıyla gerçekleştirilir.
* **Hamming (7,4) Hata Düzeltme Kodu (ECC):** Gömülecek bit dizisi 7-bitlik bloklara dönüştürülerek olası bit hataları otomatik olarak düzeltilir.     
