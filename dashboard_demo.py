import numpy as np
import pywt
import scipy.signal
from scipy.fftpack import dct, idct
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import glob

############################################
# CONSTANTS & CONFIG
############################################
FRAME_SIZE = 2048
WAVELET = 'db4'
DWT_LEVEL = 3
ALPHA = 0.5  # Gömme şiddeti

############################################
# UTILS & HELPERS
############################################
def str_to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:].zfill(8)
        result.extend([int(b) for b in bits])
    return np.array(result)

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
    corrected = data.copy()
    if syndrome != 0:
        pos_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
        if syndrome in pos_map: corrected[pos_map[syndrome]] = 1 - corrected[pos_map[syndrome]]
    return corrected[[2, 4, 5, 6]]

def hamming_decode(bits, pad):
    decoded = []
    remainder = len(bits) % 7
    if remainder != 0: bits = bits[:-remainder]
    for i in range(0, len(bits), 7):
        decoded.extend(hamming_decode_nibble(bits[i:i+7]))
    decoded = np.array(decoded)
    return decoded[:-pad] if pad > 0 else decoded

def apply_dct2(a): return dct(dct(a.T, norm='ortho').T, norm='ortho')
def apply_idct2(a): return idct(idct(a.T, norm='ortho').T, norm='ortho')

def qim_embed(val, bit, delta):
    if bit == 0: return np.round(val / delta) * delta
    else: return np.round((val - delta/2) / delta) * delta + delta/2

############################################
# WATERMARKING (UPDATED FOR DASHBOARD)
############################################
def embed_watermark(audio, msg_bits):
    pad_len = FRAME_SIZE - (len(audio) % FRAME_SIZE)
    if pad_len != FRAME_SIZE: audio = np.pad(audio, (0, pad_len))
        
    num_frames = len(audio) // FRAME_SIZE
    watermarked = audio.copy()
    msg_frame_count = min(len(msg_bits), num_frames)
    
    # Grafik için ilk frame katsayılarını sakla
    first_cA_orig = None
    first_cA_mod = None

    for i in range(msg_frame_count):
        frame = audio[i*FRAME_SIZE : (i+1)*FRAME_SIZE]
        bit = msg_bits[i]
        coeffs = pywt.wavedec(frame, WAVELET, level=DWT_LEVEL)
        cA3, cD3 = coeffs[0], coeffs[1]
        
        delta = ALPHA * np.mean(np.abs(cA3))
        if delta < 1e-6: continue
            
        rows = 16
        cols = len(cD3) // rows
        A = cD3[:rows*cols].reshape(rows, cols)
        A_dct = apply_dct2(A)
        U, S, V = np.linalg.svd(A_dct, full_matrices=False)
        
        S[0] = qim_embed(S[0], bit, delta)
        A_mod = apply_idct2(U @ np.diag(S) @ V)
        
        cD3[:rows*cols] = A_mod.flatten()
        coeffs[1] = cD3
        frame_mod = pywt.waverec(coeffs, WAVELET)[:FRAME_SIZE]
        watermarked[i*FRAME_SIZE : (i+1)*FRAME_SIZE] = frame_mod
        
        if i == 0: # İlk frame verilerini dashboard için al
            first_cA_orig = cA3
            # Modifiye edilmiş cA'yı tekrar analizle al
            first_cA_mod = pywt.wavedec(frame_mod, WAVELET, level=DWT_LEVEL)[0]
        
    return watermarked, pad_len, first_cA_orig, first_cA_mod

def extract_watermark(audio, msg_len_bits):
    num_frames = len(audio) // FRAME_SIZE
    extracted_bits = []
    for i in range(min(msg_len_bits, num_frames)):
        frame = audio[i*FRAME_SIZE : (i+1)*FRAME_SIZE]
        coeffs = pywt.wavedec(frame, WAVELET, level=DWT_LEVEL)
        cA3, cD3 = coeffs[0], coeffs[1]
        delta = ALPHA * np.mean(np.abs(cA3))
        
        rows, cols = 16, len(cD3) // 16
        A_dct = apply_dct2(cD3[:rows*cols].reshape(rows, cols))
        val = np.linalg.svd(A_dct, compute_uv=False)[0]
        
        d0 = np.abs(val - qim_embed(val, 0, delta))
        d1 = np.abs(val - qim_embed(val, 1, delta))
        extracted_bits.append(0 if d0 < d1 else 1)
    
    res = np.array(extracted_bits)
    return np.pad(res, (0, msg_len_bits - len(res))) if len(res) < msg_len_bits else res

############################################
# ATTACKS & EVAL[cite: 1]
############################################
def attack_awgn(x, snr_db):
    p_sig = np.mean(x**2)
    p_noise = p_sig / (10**(snr_db/10))
    return x + np.random.normal(0, np.sqrt(p_noise), len(x))

def attack_lpf(x, sr, cutoff):
    b, a = scipy.signal.butter(4, cutoff/(0.5*sr), btype='low')
    return scipy.signal.filtfilt(b, a, x)

def calculate_snr(orig, wm):
    p_sig, p_noise = np.sum(orig**2), np.sum((orig-wm)**2)
    return 10 * np.log10(p_sig/p_noise) if p_noise > 0 else 60.0

def calculate_ber(orig, ext):
    return np.sum(orig != ext) / len(orig)

############################################
# DASHBOARD VISUALIZATION[cite: 1]
############################################
def create_visual_dashboard(audio, watermarked, sr, attacks_results, cA_orig, cA_mod, filename):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), facecolor='#f4f4f4')
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # 1. BER Çubuğu
    names, vals = list(attacks_results.keys()), list(attacks_results.values())
    bars = axs[0, 0].bar(names, vals, color='#b22222')
    axs[0, 0].axhline(y=0.05, color='orange', linestyle='--', label='Eşik 0.05')
    axs[0, 0].set_title("BER — Saldırı Senaryoları\n(Normalize QIM + Adaptif α)", fontsize=11, fontweight='bold', color='navy')
    axs[0, 0].set_ylim(0, 0.75)
    plt.setp(axs[0, 0].get_xticklabels(), rotation=35, ha='right', fontsize=8)
    for b in bars:
        axs[0, 0].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=8)

    # 2. DWT Katsayıları
    axs[0, 1].plot(cA_orig[:128], label='Orijinal cA', color='navy', alpha=0.7)
    axs[0, 1].plot(cA_mod[:128], label='Filigranlanmış cA', color='darkred', linestyle='--', alpha=0.7)
    axs[0, 1].set_title(f"DWT Yaklaşım Katsayıları (cA{DWT_LEVEL})\nNormalize QIM ile gömme", fontsize=11, fontweight='bold', color='navy')
    axs[0, 1].legend(loc='upper right', fontsize=9)
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Fark Sinyali
    diff = (watermarked - audio) * 20
    time = np.linspace(0, (2000/sr)*1000, 2000)
    axs[1, 0].plot(time, audio[:2000], color='navy', alpha=0.4, label='Orijinal')
    axs[1, 0].plot(time, diff[:2000], color='firebrick', alpha=0.8, label='Filigran x20')
    axs[1, 0].set_title("Fark Sinyali (Gömülen Filigran x20)", fontsize=11, fontweight='bold', color='navy')
    axs[1, 0].set_xlabel("Zaman (ms)")
    axs[1, 0].legend(loc='lower right', fontsize=9)

    # 4. Güç Spektrumu (PSD)
    axs[1, 1].psd(audio, NFFT=1024, Fs=sr, color='navy', label='Orijinal', alpha=0.7)
    axs[1, 1].psd(watermarked, NFFT=1024, Fs=sr, color='darkred', label='Filigranlı', linestyle='--', alpha=0.6)
    axs[1, 1].set_title("Güç Spektrumu\n(Dikey çizgiler: DWT bant sınırları)", fontsize=11, fontweight='bold', color='navy')
    axs[1, 1].set_xlim(0, sr/2)
    axs[1, 1].legend(fontsize=9)

    fig.suptitle(f"DWT Ses Filigranı — Performans Analizi: {filename}\nNormalize QIM + Adaptif α + Hamming ECC", fontsize=15, fontweight='bold', color='navy')
    plt.savefig(f'dashboard_{filename}.png', dpi=150)
    print(f"Dashboard kaydedildi: dashboard_{filename}.png")

############################################
# MAIN[cite: 1]
############################################
def main():
    wav_files = glob.glob("**/*.wav", recursive=True)
    if not wav_files: return print("WAV dosyası bulunamadı!")
    
    selected_files = wav_files[:min(3, len(wav_files))]
    message = "GTZAN_WM"
    all_results = []

    for wav_path in selected_files:
        fname = os.path.basename(wav_path)
        print(f"--- İşleniyor: {fname} ---")
        audio, sr = librosa.load(wav_path, sr=None)
        
        msg_bits_raw = str_to_bits(message)
        msg_bits, hm_pad = hamming_encode(msg_bits_raw)
        
        # Gömme ve katsayıları alma
        wm_audio, pad_len, cA_o, cA_m = embed_watermark(audio, msg_bits)
        audio_p = np.pad(audio, (0, pad_len)) if pad_len != FRAME_SIZE else audio
        
        attacks = {
            "Saldırısız": lambda x: x,
            "AWGN (20 dB)": lambda x: attack_awgn(x, 20),
            "AWGN (30 dB)": lambda x: attack_awgn(x, 30),
            "LPF (8 kHz)": lambda x: attack_lpf(x, sr, 8000),
            "Kırpma (%5)": lambda x: x * (np.random.rand(len(x)) > 0.05),
            "Genlik x1.5": lambda x: x * 1.5
        }
        
        current_ber_results = {}
        for name, fn in attacks.items():
            att_audio = fn(wm_audio)
            ext_bits_enc = extract_watermark(att_audio, len(msg_bits))
            ext_bits = hamming_decode(ext_bits_enc, hm_pad)
            ber = calculate_ber(msg_bits_raw, ext_bits)
            current_ber_results[name] = ber
            all_results.append({"Dosya": fname, "Saldırı": name, "BER": ber})
        
        # Dashboard Üret[cite: 1]
        create_visual_dashboard(audio_p, wm_audio, sr, current_ber_results, cA_o, cA_m, fname)

    print("\nİşlem tamamlandı.")

if __name__ == "__main__":
    main()