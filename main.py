import numpy as np
import pywt
import scipy.signal
from scipy.fftpack import dct, idct
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
import sys
import glob
import os

############################################
# CONSTANTS & CONFIG
############################################
FRAME_SIZE = 2048
WAVELET = 'db4'
DWT_LEVEL = 3
ALPHA = 0.5  # Gömme şiddeti (Embedding strength) - DCT ve Hamming ile optimize edildi

############################################
# UTILS: STRING TO BITS
############################################
def str_to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:].zfill(8)
        result.extend([int(b) for b in bits])
    return np.array(result)

def bits_to_str(bits):
    s = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char_code = int("".join(str(b) for b in byte), 2)
        if 32 <= char_code <= 126:
            s += chr(char_code)
        else:
            s += "?"
    return s

############################################
# HATA DÜZELTME (HAMMING 7,4 CODE) & DCT
############################################
def hamming_encode_nibble(data):
    d1, d2, d3, d4 = data
    p1 = (d1 + d2 + d4) % 2
    p2 = (d1 + d3 + d4) % 2
    p3 = (d2 + d3 + d4) % 2
    return np.array([p1, p2, d1, p3, d2, d3, d4])

def hamming_encode(bits):
    pad = (4 - len(bits) % 4) % 4
    bits = np.pad(bits, (0, pad))
    encoded = []
    for i in range(0, len(bits), 4):
        encoded.extend(hamming_encode_nibble(bits[i:i+4]))
    return np.array(encoded), pad

def hamming_decode_nibble(data):
    p1, p2, d1, p3, d2, d3, d4 = data
    s1 = (p1 + d1 + d2 + d4) % 2
    s2 = (p2 + d1 + d3 + d4) % 2
    s3 = (p3 + d2 + d3 + d4) % 2
    syndrome = s1 * 1 + s2 * 2 + s3 * 4
    
    error_pos = -1
    if syndrome == 1: error_pos = 0
    elif syndrome == 2: error_pos = 1
    elif syndrome == 3: error_pos = 2
    elif syndrome == 4: error_pos = 3
    elif syndrome == 5: error_pos = 4
    elif syndrome == 6: error_pos = 5
    elif syndrome == 7: error_pos = 6
    
    corrected = data.copy()
    if error_pos != -1:
        corrected[error_pos] = 1 - corrected[error_pos]
        
    return corrected[[2, 4, 5, 6]]

def hamming_decode(bits, pad):
    decoded = []
    remainder = len(bits) % 7
    if remainder != 0:
        bits = bits[:-remainder]
    for i in range(0, len(bits), 7):
        decoded.extend(hamming_decode_nibble(bits[i:i+7]))
    decoded = np.array(decoded)
    if pad > 0:
        decoded = decoded[:-pad]
    return decoded

def apply_dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def apply_idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

############################################
# QIM
############################################
def qim_embed(val, bit, delta):
    if bit == 0:
        return np.round(val / delta) * delta
    else:
        return np.round((val - delta/2) / delta) * delta + delta/2

############################################
# WATERMARKING
############################################
def embed_watermark(audio, msg_bits):
    pad_len = FRAME_SIZE - (len(audio) % FRAME_SIZE)
    if pad_len != FRAME_SIZE:
        audio = np.pad(audio, (0, pad_len))
        
    num_frames = len(audio) // FRAME_SIZE
    watermarked = audio.copy()  # Geri kalan çerçeveler dokunulmaz kalır
    
    # Mesajı yalnızca bir kez göm (tekrar yok)
    msg_frame_count = min(len(msg_bits), num_frames)
    
    for i in range(msg_frame_count):
        frame = audio[i*FRAME_SIZE : (i+1)*FRAME_SIZE]
        bit = msg_bits[i]
        
        coeffs = pywt.wavedec(frame, WAVELET, level=DWT_LEVEL)
        # coeffs[0]: cA3, coeffs[1]: cD3
        cA3 = coeffs[0]
        cD3 = coeffs[1]
        
        # Psikoakustik Maskeleme Yaklaşımı
        energy_cA3 = np.mean(np.abs(cA3))
        delta = ALPHA * energy_cA3
        
        if delta < 1e-6:
            watermarked[i*FRAME_SIZE : (i+1)*FRAME_SIZE] = frame
            continue
            
        rows = 16
        cols = len(cD3) // rows
        
        A = cD3[:rows*cols].reshape(rows, cols)
        
        # --- YENİ EKLENEN: DCT HİBRİT MODELİ ---
        A_dct = apply_dct2(A)
        U, S, V = np.linalg.svd(A_dct, full_matrices=False)
        
        S[0] = qim_embed(S[0], bit, delta)
        
        A_dct_mod = U @ np.diag(S) @ V
        A_mod = apply_idct2(A_dct_mod)
        # ---------------------------------------
        
        cD3[:rows*cols] = A_mod.flatten()
        coeffs[1] = cD3
        
        frame_mod = pywt.waverec(coeffs, WAVELET)
        
        if len(frame_mod) > FRAME_SIZE:
            frame_mod = frame_mod[:FRAME_SIZE]
        elif len(frame_mod) < FRAME_SIZE:
            frame_mod = np.pad(frame_mod, (0, FRAME_SIZE - len(frame_mod)))
            
        watermarked[i*FRAME_SIZE : (i+1)*FRAME_SIZE] = frame_mod
        
    return watermarked, pad_len

def extract_watermark(audio, msg_len_bits):
    num_frames = len(audio) // FRAME_SIZE
    extracted_bits = []
    
    # Yalnızca mesaj kadar çerçeveyi oku (tekrar yok, majority voting yok)
    read_count = min(msg_len_bits, num_frames)
    
    for i in range(read_count):
        frame = audio[i*FRAME_SIZE : (i+1)*FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
            
        coeffs = pywt.wavedec(frame, WAVELET, level=DWT_LEVEL)
        cA3 = coeffs[0]
        cD3 = coeffs[1]
        
        energy_cA3 = np.mean(np.abs(cA3))
        delta = ALPHA * energy_cA3
        
        if delta < 1e-6:
            extracted_bits.append(0)
            continue
            
        rows = 16
        cols = len(cD3) // rows
        
        A = cD3[:rows*cols].reshape(rows, cols)
        
        A_dct = apply_dct2(A)
        U, S, V = np.linalg.svd(A_dct, full_matrices=False)
        val = S[0]
        
        d0 = np.abs(val - qim_embed(val, 0, delta))
        d1 = np.abs(val - qim_embed(val, 1, delta))
        
        extracted_bits.append(0 if d0 < d1 else 1)
    
    extracted_bits = np.array(extracted_bits)
    
    # Çerçeve yetmediyse sıfır doldur
    if len(extracted_bits) < msg_len_bits:
        extracted_bits = np.pad(extracted_bits, (0, msg_len_bits - len(extracted_bits)))
    
    return extracted_bits

############################################
# ATTACKS
############################################
def attack_awgn(x, snr_db):
    signal_power = np.mean(x**2)
    if signal_power == 0: return x
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(x))
    return x + noise

def attack_lpf(x, sr, cutoff):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    if norm_cutoff >= 1.0: return x
    b, a = scipy.signal.butter(4, norm_cutoff, btype='low', analog=False)
    return scipy.signal.filtfilt(b, a, x)

def attack_resample(x, sr, target_sr=16000):
    x_res = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    x_restored = librosa.resample(x_res, orig_sr=target_sr, target_sr=sr)
    if len(x_restored) > len(x):
        x_restored = x_restored[:len(x)]
    else:
        x_restored = np.pad(x_restored, (0, len(x) - len(x_restored)))
    return x_restored

def attack_crop(x, percent):
    crop_len = int(len(x) * (percent / 100))
    start = (len(x) - crop_len) // 2
    x_attacked = x.copy()
    x_attacked[start:start+crop_len] = 0
    return x_attacked

def attack_amplitude(x, scale):
    return x * scale

def attack_tsm(x, rate):
    x_stretched = librosa.effects.time_stretch(x, rate=rate)
    if len(x_stretched) > len(x):
        x_stretched = x_stretched[:len(x)]
    else:
        x_stretched = np.pad(x_stretched, (0, len(x) - len(x_stretched)))
    return x_stretched

############################################
# EVALUATION
############################################
def calculate_snr(orig, watermarked):
    noise = orig - watermarked
    signal_power = np.sum(orig**2)
    noise_power = np.sum(noise**2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def calculate_ber(orig_bits, ext_bits):
    if len(orig_bits) != len(ext_bits):
        min_len = min(len(orig_bits), len(ext_bits))
        orig_bits = orig_bits[:min_len]
        ext_bits = ext_bits[:min_len]
    return np.sum(orig_bits != ext_bits) / len(orig_bits)

############################################
# MAIN PIPELINE
############################################
def main():
    print("GTZAN Veri Seti'nden ses dosyaları aranıyor...")
    import glob
    import os
    import random
    
    wav_files = glob.glob("GTZAN/**/*.wav", recursive=True)
    if not wav_files:
        wav_files = glob.glob("**/*.wav", recursive=True)
        
    if not wav_files:
        print("HATA: Hiçbir WAV dosyası bulunamadı! Lütfen veri setini proje klasörüne çıkartın.")
        return
        
    # Rastgele 5 farklı dosya seç (Test süresini 10-15 dk civarına indirmek için ideal önerim)
    random.seed(42)
    random.shuffle(wav_files)
    selected_files = wav_files[:min(5, len(wav_files))]
    print(f"Toplam {len(selected_files)} adet ses dosyası üzerinde Optimize Edilmiş Toplu Test başlatılıyor...\n")
    
    # 5 Farklı kapasite senaryosu için mesajlar (Grafikte net bir çizgi oluşturmak için yeterli)
    messages = [
        "A",                                 # 1 Karakter (Çok Düşük Kapasite, Çok Yüksek Direnç)
        "TEST",                              # 4 Karakter
        "GTZAN_WM",                          # 8 Karakter (Standart)
        "CAPACITY_TEST_LIMIT", 
        "0123456789012345678912", #22 karakter
        "01234567890123456789123", # 23 Karakter (Fiziksel Kapasite Sınırına Yakın - 266 bit)
        #"THIS_EXCEEDS_CAPACITY_LIMIT_SINEM_OZGUN_1234567890_XYZ_ABCDEFGH_IJKLMNOPQRSTUVWXYZ_123456789"    # 92 Karakter (Sınırı Aşar, Frame yetmezliği BER zıplaması yapar)
    ]
    
    all_results = []
    
    for wav_filename in selected_files:
        print(f"--- İşleniyor: {os.path.basename(wav_filename)} ---")
        audio, sr = librosa.load(wav_filename, sr=None)
        audio_duration = len(audio) / sr if sr > 0 else 1
        
        for message in messages:
            msg_bits_raw = str_to_bits(message)
            msg_bits, hm_pad = hamming_encode(msg_bits_raw)
            
            capacity_bps = len(msg_bits_raw) / audio_duration
            
            watermarked, pad_len = embed_watermark(audio, msg_bits)
            
            if pad_len != FRAME_SIZE:
                audio_padded = np.pad(audio, (0, pad_len))
            else:
                audio_padded = audio
                
            snr_val = calculate_snr(audio_padded, watermarked)
            snr_val_clean = snr_val if snr_val != float('inf') else 60.0
            
            attacks = {
                "No Attack": lambda x: x,
                "AWGN 30dB": lambda x: attack_awgn(x, 30),
                "AWGN 20dB": lambda x: attack_awgn(x, 20),
                "LPF 12kHz": lambda x: attack_lpf(x, sr, 12000),
                "LPF 8kHz": lambda x: attack_lpf(x, sr, 8000),
                "Resampling 16kHz": lambda x: attack_resample(x, sr, 16000),
                "Cropping %5": lambda x: attack_crop(x, 5),
                "Cropping %10": lambda x: attack_crop(x, 10),
                "Amplitude x2": lambda x: attack_amplitude(x, 2.0),
                "Amplitude x0.5": lambda x: attack_amplitude(x, 0.5),
                "TSM +%5": lambda x: attack_tsm(x, 1.05),
                "TSM -%5": lambda x: attack_tsm(x, 0.95),
            }
            
            for name, attack_fn in attacks.items():
                attacked_audio = attack_fn(watermarked)
                extracted_encoded_bits = extract_watermark(attacked_audio, len(msg_bits))
                extracted_bits = hamming_decode(extracted_encoded_bits, hm_pad)
                ber = calculate_ber(msg_bits_raw, extracted_bits)
                
                all_results.append({
                    "Dosya": os.path.basename(wav_filename),
                    "Mesaj_Karakter_Sayısı": len(message),
                    "Mesaj": message,
                    "Kapasite_bps": capacity_bps,
                    "Saldırı_Türü": name,
                    "BER": ber,
                    "SNR_dB": snr_val_clean
                })
                
    # Verileri Analiz Et ve Kaydet
    df = pd.DataFrame(all_results)
    df.to_csv("batch_results.csv", index=False)
    print("\nBatch test tamamlandı. Tüm sonuçlar 'batch_results.csv' dosyasına kaydedildi.")
    
    # 1. Ortalama BER vs Saldırı Türü Grafiği
    avg_ber_by_attack = df.groupby("Saldırı_Türü")["BER"].mean().sort_values()
    plt.figure(figsize=(12, 6))
    avg_ber_by_attack.plot(kind='bar', color='coral')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Ortalama Bit Error Rate (BER)')
    plt.title('Tüm Dosya ve Mesajlar Üzerinden Ortalama Saldırı Hasarı')
    plt.tight_layout()
    plt.savefig('avg_ber_attacks.png')
    print("Grafik kaydedildi: avg_ber_attacks.png")
    
    # 2. Mesaj Uzunluğu vs Ortalama BER Grafiği (Kapasite vs Dayanıklılık)
    harsh_attacks = df[df["Saldırı_Türü"].isin(["AWGN 20dB", "Cropping %10", "TSM +%5"])]
    avg_ber_by_length = harsh_attacks.groupby("Mesaj_Karakter_Sayısı")["BER"].mean()
    
    plt.figure(figsize=(10, 5))
    avg_ber_by_length.plot(kind='line', marker='o', color='purple', linewidth=2)
    plt.xticks(avg_ber_by_length.index)
    plt.xlabel('Gizlenen Mesaj Uzunluğu (Karakter Sayısı)')
    plt.ylabel('Ortalama BER (Ağır Saldırılar: AWGN20, Crop10, TSM)')
    plt.title('Kapasite vs Dayanıklılık (Capacity vs Robustness)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('ber_vs_payload.png')
    print("Grafik kaydedildi: ber_vs_payload.png")
    
    # 3. Mesaj Uzunluğu vs SNR Grafiği (Algılanamazlık ve Tespit Edilemezlik)
    # Gömme işlemi sonucu elde edilen orijinal ses ile damgalı ses arasındaki ortalama SNR
    avg_snr_by_length = df.groupby("Mesaj_Karakter_Sayısı")["SNR_dB"].mean()
    
    plt.figure(figsize=(10, 5))
    avg_snr_by_length.plot(kind='line', marker='s', color='green', linewidth=2)
    plt.axhline(y=30, color='red', linestyle='--', linewidth=1, label="Duyulmazlık Eşiği (30 dB)")
    plt.xticks(avg_snr_by_length.index)
    plt.xlabel('Gizlenen Mesaj Uzunluğu (Karakter Sayısı)')
    plt.ylabel('Ortalama SNR (dB)')
    plt.title('Kapasite vs Algılanamazlık (Imperceptibility & Undetectability)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('snr_vs_payload.png')
    print("Grafik kaydedildi: snr_vs_payload.png")
    
    # Genel Rapor Özeti
    print("\n================ ÖZET RAPOR ================")
    print(f"Toplam Test Edilen Dosya: {len(selected_files)}")
    print(f"Toplam Test Edilen Mesaj Sayısı: {len(messages)}")
    print(f"Ortalama Algılanamazlık (SNR): {df['SNR_dB'].mean():.2f} dB")
    print("============================================")

if __name__ == "__main__":
    main()