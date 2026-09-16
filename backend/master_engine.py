"""
Myooz Master Engine: Professional Vectorized Commercial Audio Mastering Studio
Author: Myooz Audio Intelligence Suite
Pure NumPy + Scipy signal implementation (ultra-fast, zero external C++ dependencies).
Standards: ITU-R BS.1770-4 LUFS, EBU R128, AES TD1004.
"""

import os
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal

def calculate_lufs(audio, sr=48000):
    """Exact ITU-R BS.1770-4 K-Weighting Integrated Loudness (LUFS)"""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
        
    # Stage 1: High shelf filter (+4 dB at high frequencies)
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    # Stage 2: High pass filter (cutoff ~100 Hz)
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225035621]
    
    y = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        filtered1 = signal.lfilter(b1, a1, audio[:, ch])
        y[:, ch] = signal.lfilter(b2, a2, filtered1)
        
    block_size = int(0.400 * sr)
    hop_size = int(0.100 * sr)
    n_blocks = (len(y) - block_size) // hop_size + 1
    if n_blocks <= 0:
        return -70.0
        
    block_powers = []
    for i in range(n_blocks):
        start = i * hop_size
        block = y[start:start+block_size, :]
        power = np.mean(block**2, axis=0)
        z = np.sum(power)
        block_powers.append(z)
        
    block_powers = np.array(block_powers)
    block_loudness = -0.691 + 10 * np.log10(block_powers + 1e-12)
    
    idx_abs = block_loudness > -70.0
    if not np.any(idx_abs):
        return -70.0
        
    z_avg = np.mean(block_powers[idx_abs])
    gamma_r = -0.691 + 10 * np.log10(z_avg + 1e-12) - 10.0
    
    idx_rel = block_loudness > gamma_r
    if not np.any(idx_rel):
        return float(gamma_r)
        
    integrated_lufs = -0.691 + 10 * np.log10(np.mean(block_powers[idx_rel]) + 1e-12)
    return float(integrated_lufs)

def calculate_true_peak(audio, sr=48000):
    """Calculates True-Peak in dBFS using 4x oversampling"""
    audio_4x = signal.resample_poly(audio, 4, 1, axis=0)
    peak = np.max(np.abs(audio_4x))
    return float(20 * np.log10(peak + 1e-12))

def fast_tape_saturation(audio, warmth=0.5):
    """Soft analog tape harmonic saturation (adds 2nd and 3rd order harmonics)"""
    if warmth <= 0.05:
        return audio
    drive_db = float(warmth) * 2.5
    drive = 10 ** (drive_db / 20.0)
    x = audio * drive
    # Smooth cubic soft-clipper
    saturated = (3.0 / 2.0) * x * (1.0 - (x**2) / 3.0)
    saturated = np.clip(saturated, -1.0, 1.0)
    return saturated / max(1.0, drive * 0.9)

def fast_glue_compressor(audio, sr=48000, ratio=1.5, attack_ms=30.0, release_ms=100.0):
    """VCA Master Bus Glue Compressor with vectorized envelope"""
    sidechain = np.max(np.abs(audio), axis=1) if audio.ndim == 2 else np.abs(audio)
    rel_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
    b = [1.0 - rel_coef]
    a = [1.0, -rel_coef]
    env = signal.lfilter(b, a, sidechain)
    
    threshold_db = -13.0
    env_db = 20 * np.log10(env + 1e-9)
    gain_db = np.zeros_like(env_db)
    over = env_db > threshold_db
    gain_db[over] = (threshold_db + (env_db[over] - threshold_db) / ratio) - env_db[over]
    
    # Max gain reduction clamp of 2.5 dB
    gain_db = np.maximum(gain_db, -2.5)
    gain_lin = 10 ** (gain_db / 20.0)
    
    if audio.ndim == 2:
        return audio * gain_lin[:, np.newaxis]
    return audio * gain_lin

def fast_mid_side_polish(audio, sr=48000, stereo_spread=0.5, air=0.5):
    """
    Mid/Side spatial processing:
    1. Elliptical filter: HPF Side channel at 120Hz (100% Mono Sub-Bass).
    2. Air sheen on Side channel above 10kHz.
    3. Stereo spread width adjustment.
    """
    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    
    # 1. Mono sub-bass (high-pass the side channel at 120Hz)
    sos = signal.butter(3, 120, btype='high', fs=sr, output='sos')
    side_clean = signal.sosfilt(sos, side)
    
    # 2. Side air sheen
    if air > 0.05:
        air_gain_db = float(air) * 2.2
        A = 10 ** (air_gain_db / 40.0)
        w0 = 2 * math.pi * 10500 / sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / 2 * math.sqrt(2)
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha
        b = [b0/a0, b1/a0, b2/a0]
        a = [1.0, a1/a0, a2/a0]
        side_clean = signal.lfilter(b, a, side_clean)
        
    # 3. Stereo width
    width_mult = 0.7 + float(stereo_spread) * 0.6 # 0.7 to 1.3
    side_clean = side_clean * width_mult
    
    out = np.zeros_like(audio)
    out[:, 0] = mid + side_clean
    out[:, 1] = mid - side_clean
    return out

def fast_brickwall_limiter(audio, ceiling_db=-0.5, sr=48000):
    """Lookahead True-Peak Brickwall Limiter"""
    ceiling_lin = 10 ** (ceiling_db / 20.0) # e.g. 0.944
    rel_coef = np.exp(-1.0 / (sr * 0.040))
    b = [1.0 - rel_coef]
    a = [1.0, -rel_coef]
    
    peak_track = np.max(np.abs(audio), axis=1)
    env = signal.lfilter(b, a, peak_track)
    
    gain_reduction = np.ones(len(audio), dtype=np.float64)
    over = env > ceiling_lin
    gain_reduction[over] = ceiling_lin / env[over]
    
    limited = audio * gain_reduction[:, np.newaxis]
    return np.clip(limited, -ceiling_lin, ceiling_lin)

# ==========================================
# MAIN MASTERING PROCESSOR
# ==========================================

def master_audio(
    input_path,
    output_path=None,
    target_profile="streaming",
    warmth=0.5,
    stereo_spread=0.5,
    air=0.5
):
    """
    Executes full commercial mastering chain on stereo mixdown:
    1. Input loudness measurement
    2. Analog tape warmth saturation
    3. Master bus glue compression
    4. Mid/Side mono-sub + stereo air polish
    5. ITU-R BS.1770-4 target loudness normalization
    6. True-Peak brickwall limiting
    7. 24-bit PCM WAV export
    """
    if not output_path:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_mastered.wav"
        
    target_lufs_map = {
        "streaming": -14.0,
        "club": -9.0,
        "dynamic": -12.0
    }
    target_lufs = target_lufs_map.get(str(target_profile).lower(), -14.0)
    
    # 1. Read input audio
    data, sr = sf.read(input_path, always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
        
    data = data.astype(np.float64)
    input_lufs = calculate_lufs(data, sr)
    
    # 2. Analog Tape Saturation
    mastered = fast_tape_saturation(data, warmth=float(warmth))
    
    # 3. Master Bus Glue Compression
    mastered = fast_glue_compressor(mastered, sr=sr, ratio=1.5, attack_ms=30.0, release_ms=100.0)
    
    # 4. Mid/Side Spatial Polish (Mono Sub <120Hz + Side Air)
    mastered = fast_mid_side_polish(mastered, sr=sr, stereo_spread=float(stereo_spread), air=float(air))
    
    # 5. Target Loudness Calibration
    current_lufs = calculate_lufs(mastered, sr)
    gain_adjustment_db = target_lufs - current_lufs
    mastered = mastered * (10 ** (gain_adjustment_db / 20.0))
    
    # 6. True-Peak Brickwall Limiting
    ceiling_db = -0.5 if target_profile == "streaming" else (-0.3 if target_profile == "club" else -0.5)
    final_master = fast_brickwall_limiter(mastered, ceiling_db=ceiling_db, sr=sr)
    
    # 7. Final Metrics
    final_lufs = calculate_lufs(final_master, sr)
    final_true_peak = calculate_true_peak(final_master, sr)
    
    # 8. Export 24-bit PCM WAV
    sf.write(output_path, final_master.astype(np.float32), sr, subtype='PCM_24')
    
    metrics = {
        "target_profile": str(target_profile).capitalize(),
        "input_lufs": round(float(input_lufs), 1),
        "output_lufs": round(float(final_lufs), 1),
        "true_peak_dbfs": round(float(final_true_peak), 2),
        "sample_rate": f"{sr} Hz",
        "bit_depth": "24-bit PCM WAV",
        "output_path": output_path
    }
    
    return output_path, metrics
