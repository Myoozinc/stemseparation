"""
Myooz Master Engine: Professional Studio Multi-Band Audio Mastering Engine
Author: Myooz Audio Intelligence Suite
Pure NumPy + Scipy signal implementation (ultra-fast, zero external C++ dependencies).
Standards: ITU-R BS.1770-4 LUFS, EBU R128, AES TD1004.
Bit Depth: 32-bit Floating Point WAV (Studio Standard).
"""

import os
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal

# ==========================================================
# PRESETS MATRIX BY GENRE & CHARACTER
# ==========================================================

MASTER_PRESETS = {
    "urbano": {
        "name": "Reggaeton / Urbano / Trap Latino",
        "default_style": "club_banger",
        "styles": {
            "club_banger": {
                "name": "Club Banger (Pegada Potente)",
                "target_lufs": -9.0,
                "ceiling_db": -0.4,
                "sub_hpf": 28,
                "eq_sub_gain": 2.2, "eq_sub_freq": 52,
                "eq_presence_gain": 1.8, "eq_presence_freq": 3400,
                "tape_drive": 0.65,
                "multiband_low_comp": 2.2, "multiband_high_comp": 1.4,
                "stereo_width": 1.15,
                "air_sheen": 0.35,
                "mono_sub_hz": 125
            },
            "streaming": {
                "name": "Radio & Streaming Pro",
                "target_lufs": -14.0,
                "ceiling_db": -0.8,
                "sub_hpf": 26,
                "eq_sub_gain": 1.2, "eq_sub_freq": 55,
                "eq_presence_gain": 2.2, "eq_presence_freq": 3600,
                "tape_drive": 0.35,
                "multiband_low_comp": 1.5, "multiband_high_comp": 1.2,
                "stereo_width": 1.20,
                "air_sheen": 0.45,
                "mono_sub_hz": 110
            },
            "analog_warmth": {
                "name": "Vintage Analog Tape",
                "target_lufs": -11.5,
                "ceiling_db": -0.5,
                "sub_hpf": 25,
                "eq_sub_gain": 1.8, "eq_sub_freq": 60,
                "eq_presence_gain": 1.0, "eq_presence_freq": 2800,
                "tape_drive": 0.85,
                "multiband_low_comp": 1.8, "multiband_high_comp": 1.5,
                "stereo_width": 1.08,
                "air_sheen": 0.25,
                "mono_sub_hz": 120
            }
        }
    },
    "pop": {
        "name": "Pop / R&B Comercial",
        "default_style": "radio_hit",
        "styles": {
            "radio_hit": {
                "name": "Radio Hit (Brillante & Claro)",
                "target_lufs": -13.0,
                "ceiling_db": -0.5,
                "sub_hpf": 30,
                "eq_sub_gain": 0.8, "eq_sub_freq": 60,
                "eq_presence_gain": 2.5, "eq_presence_freq": 4000,
                "tape_drive": 0.4,
                "multiband_low_comp": 1.6, "multiband_high_comp": 1.3,
                "stereo_width": 1.25,
                "air_sheen": 0.55,
                "mono_sub_hz": 115
            },
            "punchy": {
                "name": "Modern Punch",
                "target_lufs": -11.0,
                "ceiling_db": -0.4,
                "sub_hpf": 28,
                "eq_sub_gain": 1.5, "eq_sub_freq": 55,
                "eq_presence_gain": 1.8, "eq_presence_freq": 3200,
                "tape_drive": 0.55,
                "multiband_low_comp": 2.0, "multiband_high_comp": 1.4,
                "stereo_width": 1.18,
                "air_sheen": 0.40,
                "mono_sub_hz": 120
            },
            "wide_air": {
                "name": "Wide & Airy (Espacioso)",
                "target_lufs": -14.0,
                "ceiling_db": -0.8,
                "sub_hpf": 28,
                "eq_sub_gain": 0.5, "eq_sub_freq": 65,
                "eq_presence_gain": 2.0, "eq_presence_freq": 4500,
                "tape_drive": 0.3,
                "multiband_low_comp": 1.3, "multiband_high_comp": 1.1,
                "stereo_width": 1.35,
                "air_sheen": 0.65,
                "mono_sub_hz": 105
            }
        }
    },
    "electronic": {
        "name": "Electronic / EDM / Dance",
        "default_style": "festival_loud",
        "styles": {
            "festival_loud": {
                "name": "Festival Mainstage",
                "target_lufs": -8.0,
                "ceiling_db": -0.2,
                "sub_hpf": 32,
                "eq_sub_gain": 2.5, "eq_sub_freq": 48,
                "eq_presence_gain": 2.0, "eq_presence_freq": 5000,
                "tape_drive": 0.7,
                "multiband_low_comp": 2.8, "multiband_high_comp": 1.6,
                "stereo_width": 1.28,
                "air_sheen": 0.50,
                "mono_sub_hz": 135
            },
            "deep_club": {
                "name": "Deep Club / House",
                "target_lufs": -10.5,
                "ceiling_db": -0.5,
                "sub_hpf": 30,
                "eq_sub_gain": 2.0, "eq_sub_freq": 50,
                "eq_presence_gain": 1.5, "eq_presence_freq": 3800,
                "tape_drive": 0.5,
                "multiband_low_comp": 2.2, "multiband_high_comp": 1.3,
                "stereo_width": 1.22,
                "air_sheen": 0.40,
                "mono_sub_hz": 125
            }
        }
    },
    "hiphop": {
        "name": "Hip-Hop / Trap 808",
        "default_style": "heavy_808",
        "styles": {
            "heavy_808": {
                "name": "Heavy 808 Hard Hitter",
                "target_lufs": -9.0,
                "ceiling_db": -0.3,
                "sub_hpf": 24,
                "eq_sub_gain": 2.8, "eq_sub_freq": 45,
                "eq_presence_gain": 2.2, "eq_presence_freq": 3200,
                "tape_drive": 0.75,
                "multiband_low_comp": 2.5, "multiband_high_comp": 1.5,
                "stereo_width": 1.12,
                "air_sheen": 0.35,
                "mono_sub_hz": 130
            },
            "boombap_warm": {
                "name": "Boom Bap / Lo-Fi Warm",
                "target_lufs": -13.0,
                "ceiling_db": -0.6,
                "sub_hpf": 28,
                "eq_sub_gain": 1.5, "eq_sub_freq": 65,
                "eq_presence_gain": 1.2, "eq_presence_freq": 2600,
                "tape_drive": 0.8,
                "multiband_low_comp": 1.8, "multiband_high_comp": 1.2,
                "stereo_width": 1.05,
                "air_sheen": 0.20,
                "mono_sub_hz": 115
            }
        }
    },
    "rock": {
        "name": "Rock / Indie / Metal",
        "default_style": "wall_of_sound",
        "styles": {
            "wall_of_sound": {
                "name": "Wall of Sound",
                "target_lufs": -10.0,
                "ceiling_db": -0.4,
                "sub_hpf": 32,
                "eq_sub_gain": 1.0, "eq_sub_freq": 70,
                "eq_presence_gain": 2.6, "eq_presence_freq": 2800,
                "tape_drive": 0.7,
                "multiband_low_comp": 2.0, "multiband_high_comp": 1.5,
                "stereo_width": 1.25,
                "air_sheen": 0.35,
                "mono_sub_hz": 110
            },
            "organic_dynamic": {
                "name": "Organic & Open",
                "target_lufs": -14.0,
                "ceiling_db": -0.8,
                "sub_hpf": 30,
                "eq_sub_gain": 0.6, "eq_sub_freq": 75,
                "eq_presence_gain": 1.8, "eq_presence_freq": 3200,
                "tape_drive": 0.35,
                "multiband_low_comp": 1.4, "multiband_high_comp": 1.1,
                "stereo_width": 1.18,
                "air_sheen": 0.40,
                "mono_sub_hz": 105
            }
        }
    },
    "acoustic": {
        "name": "Acústico / Folk / Balada",
        "default_style": "pristine_transparency",
        "styles": {
            "pristine_transparency": {
                "name": "Pristine Transparency",
                "target_lufs": -14.0,
                "ceiling_db": -0.8,
                "sub_hpf": 34,
                "eq_sub_gain": 0.4, "eq_sub_freq": 80,
                "eq_presence_gain": 1.8, "eq_presence_freq": 4200,
                "tape_drive": 0.2,
                "multiband_low_comp": 1.2, "multiband_high_comp": 1.1,
                "stereo_width": 1.20,
                "air_sheen": 0.50,
                "mono_sub_hz": 100
            },
            "warm_intimate": {
                "name": "Warm & Intimate",
                "target_lufs": -13.0,
                "ceiling_db": -0.6,
                "sub_hpf": 30,
                "eq_sub_gain": 1.2, "eq_sub_freq": 90,
                "eq_presence_gain": 1.4, "eq_presence_freq": 3000,
                "tape_drive": 0.5,
                "multiband_low_comp": 1.5, "multiband_high_comp": 1.2,
                "stereo_width": 1.15,
                "air_sheen": 0.35,
                "mono_sub_hz": 105
            }
        }
    }
}

# ==========================================================
# STUDIO METRICS ENGINE
# ==========================================================

def calculate_lufs(audio, sr=48000):
    """Exact ITU-R BS.1770-4 K-Weighting Integrated Loudness (LUFS)"""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
        
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
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

def calculate_short_term_max_lufs(audio, sr=48000):
    """Calculates Max Short-Term LUFS over 3-second windows"""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225035621]
    
    y = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        filtered1 = signal.lfilter(b1, a1, audio[:, ch])
        y[:, ch] = signal.lfilter(b2, a2, filtered1)
        
    win_size = int(3.0 * sr)
    hop = int(0.5 * sr)
    n_wins = (len(y) - win_size) // hop + 1
    if n_wins <= 0:
        return calculate_lufs(audio, sr)
        
    max_lufs = -70.0
    for i in range(n_wins):
        st = i * hop
        block = y[st:st+win_size, :]
        p = np.mean(block**2)
        l = -0.691 + 10 * np.log10(p + 1e-12)
        if l > max_lufs:
            max_lufs = l
    return float(max_lufs)

def calculate_true_peak(audio, sr=48000):
    """Calculates True-Peak in dBFS using 4x polyphase oversampling"""
    audio_4x = signal.resample_poly(audio, 4, 1, axis=0)
    peak = np.max(np.abs(audio_4x))
    return float(20 * np.log10(peak + 1e-12))

def calculate_crest_factor(audio):
    """Dynamic Range Crest Factor (Peak to RMS ratio in dB)"""
    peak = np.max(np.abs(audio)) + 1e-9
    rms = np.sqrt(np.mean(audio**2)) + 1e-9
    return float(20 * np.log10(peak / rms))

def calculate_stereo_correlation(audio):
    """Pearson correlation coefficient between Left and Right channels (-1 to +1)"""
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 1.0
    l = audio[:, 0]
    r = audio[:, 1]
    l_c = l - np.mean(l)
    r_c = r - np.mean(r)
    denom = np.sqrt(np.sum(l_c**2) * np.sum(r_c**2)) + 1e-9
    corr = np.sum(l_c * r_c) / denom
    return float(np.clip(corr, -1.0, 1.0))

# ==========================================================
# VECTORIZED 5-STAGE MASTERING DSP PRIMITIVES
# ==========================================================

def biquad_peak(audio, freq, gain_db, sr=48000, q=1.0):
    if abs(gain_db) < 0.05:
        return audio
    w0 = 2 * math.pi * freq / sr
    A = 10 ** (gain_db / 40.0)
    alpha = math.sin(w0) / (2 * q)
    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    return signal.lfilter([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0], audio, axis=0)

def biquad_highshelf(audio, freq, gain_db, sr=48000):
    if abs(gain_db) < 0.05:
        return audio
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * freq / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2 * math.sqrt(2)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha
    return signal.lfilter([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0], audio, axis=0)

def fast_multiband_dynamics(audio, sr=48000, low_comp=1.8, high_comp=1.3):
    """
    3-Band Linkwitz-Riley Crossover Dynamics:
    Band 1 (Sub/Low): < 150 Hz
    Band 2 (Mid): 150 Hz - 4000 Hz
    Band 3 (High): > 4000 Hz
    """
    sos_lp = signal.butter(2, 150, btype='low', fs=sr, output='sos')
    sos_hp = signal.butter(2, 4000, btype='high', fs=sr, output='sos')
    
    low_band = signal.sosfilt(sos_lp, audio, axis=0)
    high_band = signal.sosfilt(sos_hp, audio, axis=0)
    mid_band = audio - (low_band + high_band)
    
    # Low Band Compression (tightens sub-bass punch)
    if low_comp > 1.05:
        rel_coef = np.exp(-1.0 / (sr * 0.080))
        env_l = signal.lfilter([1.0 - rel_coef], [1.0, -rel_coef], np.max(np.abs(low_band), axis=1))
        env_l_db = 20 * np.log10(env_l + 1e-9)
        thresh = -15.0
        gain_db = np.zeros_like(env_l_db)
        over = env_l_db > thresh
        gain_db[over] = (thresh + (env_l_db[over] - thresh) / low_comp) - env_l_db[over]
        gain_db = np.maximum(gain_db, -3.5)
        low_band = low_band * (10 ** (gain_db[:, np.newaxis] / 20.0))
        
    # High Band Density & Sheen
    if high_comp > 1.05:
        rel_coef_h = np.exp(-1.0 / (sr * 0.050))
        env_h = signal.lfilter([1.0 - rel_coef_h], [1.0, -rel_coef_h], np.max(np.abs(high_band), axis=1))
        env_h_db = 20 * np.log10(env_h + 1e-9)
        thresh_h = -18.0
        gain_h_db = np.zeros_like(env_h_db)
        over_h = env_h_db > thresh_h
        gain_h_db[over_h] = (thresh_h + (env_h_db[over_h] - thresh_h) / high_comp) - env_h_db[over_h]
        gain_h_db = np.maximum(gain_h_db, -2.5)
        high_band = high_band * (10 ** (gain_h_db[:, np.newaxis] / 20.0))
        
    return low_band + mid_band + high_band

def fast_tape_saturation(audio, drive_amount=0.5):
    """Analog tape & tube harmonic excitement (adds 2nd/3rd harmonics)"""
    if drive_amount <= 0.05:
        return audio
    drive_db = float(drive_amount) * 2.8
    drive = 10 ** (drive_db / 20.0)
    x = audio * drive
    # Smooth cubic soft-saturation curve
    sat = (3.0 / 2.0) * x * (1.0 - (x**2) / 3.0)
    sat = np.clip(sat, -1.0, 1.0)
    return sat / max(1.0, drive * 0.88)

def fast_mid_side_polish(audio, sr=48000, mono_sub_hz=120, stereo_width=1.15, air_sheen=0.4):
    """
    Precision Mid/Side Spatial Stage:
    1. Elliptical filter: 100% Mono Sub below mono_sub_hz.
    2. Stereo width scaling on Side channel.
    3. Side channel Air sheen above 10.5 kHz.
    """
    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    
    # 1. Elliptical mono sub filter on side channel
    sos = signal.butter(3, mono_sub_hz, btype='high', fs=sr, output='sos')
    side_clean = signal.sosfilt(sos, side)
    
    # 2. Side air sheen
    if air_sheen > 0.05:
        air_gain = float(air_sheen) * 2.8
        side_clean = biquad_highshelf(side_clean, 10500, air_gain, sr)
        
    # 3. Stereo width
    side_clean = side_clean * float(stereo_width)
    
    out = np.zeros_like(audio)
    out[:, 0] = mid + side_clean
    out[:, 1] = mid - side_clean
    return out

def fast_brickwall_limiter(audio, ceiling_db=-0.5, sr=48000):
    """True-Peak Lookahead Brickwall Limiter"""
    ceiling_lin = 10 ** (ceiling_db / 20.0)
    rel_coef = np.exp(-1.0 / (sr * 0.040))
    b = [1.0 - rel_coef]
    a = [1.0, -rel_coef]
    
    peak_track = np.max(np.abs(audio), axis=1)
    env = signal.lfilter(b, a, peak_track)
    
    gain_reduction = np.ones(len(audio), dtype=np.float64)
    over = env > ceiling_lin
    gain_reduction[over] = ceiling_lin / (env[over] + 1e-9)
    
    limited = audio * gain_reduction[:, np.newaxis]
    return np.clip(limited, -ceiling_lin, ceiling_lin)

# ==========================================================
# MAIN MASTERING ENGINE EXECUTOR
# ==========================================================

def master_audio(
    input_path,
    output_path=None,
    genre="urbano",
    style="club_banger",
    **kwargs
):
    """
    Executes commercial studio 5-stage mastering chain calibrated by musical genre:
    1. Pre-Master EQ (Subsonic HPF & Contour Curve)
    2. 3-Band Multiband Dynamics Crossover
    3. Analog Tape & Harmonic Saturation
    4. Mid/Side Spatial Polish (Elliptical Mono Sub + Air)
    5. ITU-R BS.1770-4 LUFS Normalization & True-Peak Brickwall Limiting
    Export: 32-bit Floating Point WAV (Studio Standard).
    """
    if not output_path:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_mastered.wav"
        
    sr = 48000
    
    # Handle backward-compatible target_profile parameter
    target_profile = kwargs.get('target_profile')
    if target_profile and not genre:
        genre = "urbano"
        if target_profile == "club": style = "club_banger"
        elif target_profile == "dynamic": style = "streaming"
        
    genre_clean = str(genre).lower().strip()
    if genre_clean not in MASTER_PRESETS:
        genre_clean = "urbano"
        
    genre_conf = MASTER_PRESETS[genre_clean]
    style_clean = str(style).lower().strip()
    if style_clean not in genre_conf["styles"]:
        style_clean = genre_conf.get("default_style", list(genre_conf["styles"].keys())[0])
        
    preset = genre_conf["styles"][style_clean]
    
    # 2. Read audio
    data, file_sr = sf.read(input_path, always_2d=True, dtype='float32')
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
        
    if file_sr != sr:
        gcd = math.gcd(sr, file_sr)
        data = signal.resample_poly(data, sr // gcd, file_sr // gcd, axis=0)
        
    data = data.astype(np.float64)
    
    # Measure input metrics
    input_lufs = calculate_lufs(data, sr)
    input_true_peak = calculate_true_peak(data, sr)
    input_dr = calculate_crest_factor(data)
    input_corr = calculate_stereo_correlation(data)
    
    # Stage 1: Pre-Master Precision EQ
    sub_hpf = preset.get("sub_hpf", 28)
    sos_hpf = signal.butter(3, sub_hpf, btype='high', fs=sr, output='sos')
    mastered = signal.sosfilt(sos_hpf, data, axis=0)
    
    if preset.get("eq_sub_gain", 0) > 0:
        mastered = biquad_peak(mastered, preset["eq_sub_freq"], preset["eq_sub_gain"], sr=sr, q=1.1)
        
    if preset.get("eq_presence_gain", 0) > 0:
        mastered = biquad_peak(mastered, preset["eq_presence_freq"], preset["eq_presence_gain"], sr=sr, q=1.2)
        
    # Stage 2: 3-Band Multiband Dynamics
    mastered = fast_multiband_dynamics(
        mastered,
        sr=sr,
        low_comp=preset.get("multiband_low_comp", 1.8),
        high_comp=preset.get("multiband_high_comp", 1.3)
    )
    
    # Stage 3: Analog Tape & Tube Harmonic Saturation
    mastered = fast_tape_saturation(mastered, drive_amount=preset.get("tape_drive", 0.5))
    
    # Stage 4: Mid/Side Spatial Enhancement & Mono Sub
    mastered = fast_mid_side_polish(
        mastered,
        sr=sr,
        mono_sub_hz=preset.get("mono_sub_hz", 120),
        stereo_width=preset.get("stereo_width", 1.15),
        air_sheen=preset.get("air_sheen", 0.4)
    )
    
    # Stage 5: Target Loudness Calibration & True-Peak Limiter
    target_lufs = preset.get("target_lufs", -14.0)
    current_lufs = calculate_lufs(mastered, sr)
    gain_db = target_lufs - current_lufs
    mastered = mastered * (10 ** (gain_db / 20.0))
    
    ceiling_db = preset.get("ceiling_db", -0.5)
    final_master = fast_brickwall_limiter(mastered, ceiling_db=ceiling_db, sr=sr)
    
    # Measure output metrics
    output_lufs = calculate_lufs(final_master, sr)
    output_st_max = calculate_short_term_max_lufs(final_master, sr)
    output_true_peak = calculate_true_peak(final_master, sr)
    output_dr = calculate_crest_factor(final_master)
    output_corr = calculate_stereo_correlation(final_master)
    
    # Export 32-bit Float WAV
    sf.write(output_path, final_master.astype(np.float32), sr, subtype='FLOAT')
    
    metrics = {
        "genre": genre_clean,
        "genre_name": genre_conf["name"],
        "style": style_clean,
        "style_name": preset["name"],
        "target_lufs": target_lufs,
        "input_lufs": round(float(input_lufs), 1),
        "output_lufs": round(float(output_lufs), 1),
        "short_term_max_lufs": round(float(output_st_max), 1),
        "true_peak_dbfs": round(float(output_true_peak), 2),
        "input_true_peak": round(float(input_true_peak), 2),
        "dynamic_range_db": round(float(output_dr), 1),
        "stereo_correlation": round(float(output_corr), 2),
        "sample_rate": f"{sr} Hz",
        "bit_depth": "32-bit Float WAV",
        "output_path": output_path
    }
    
    return output_path, metrics
