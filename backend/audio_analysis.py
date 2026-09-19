"""
Myooz Audio Analysis Engine: Adaptive Signal Intelligence
Standards: ITU-R BS.1770-4, EBU R128, AES TD1004.
Pure NumPy + SciPy signal implementation (ultra-fast, zero external C++ dependencies).
Supports optional pyloudnorm when installed.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.signal as signal

# ==========================================================
# 1. LOUDNESS MEASUREMENT (ITU-R BS.1770-4 / EBU R128)
# ==========================================================

def calculate_lufs(audio: np.ndarray, sr: int = 44100) -> float:
    """
    Computes Integrated Loudness (LUFS) adhering strictly to ITU-R BS.1770-4.
    Uses K-weighting stage 1 (high shelf) + stage 2 (RLB highpass),
    followed by gating (-70 LUFS absolute gate, -10 LUFS relative gate).
    """
    if audio is None or len(audio) == 0:
        return -70.0

    # Ensure 2D array (samples, channels)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Pre-check silence
    peak = np.max(np.abs(audio))
    if peak < 1e-7:
        return -70.0

    # Optional pyloudnorm fallback/verification if available
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        if not np.isneginf(loudness) and not np.isnan(loudness):
            return round(float(loudness), 1)
    except Exception:
        pass

    # Pure SciPy ITU-R BS.1770-4 K-Weighting Filter Coefficients
    # Stage 1: High shelf filter (simulates head acoustic effects)
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]

    # Stage 2: Highpass filter (RLB weighting curve)
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225035621]

    n_samples, n_channels = audio.shape
    y = np.zeros_like(audio, dtype=np.float64)

    for ch in range(n_channels):
        f1 = signal.lfilter(b1, a1, audio[:, ch].astype(np.float64))
        y[:, ch] = signal.lfilter(b2, a2, f1)

    # 400ms rectangular gating window with 75% overlap (100ms hop)
    block_size = int(0.400 * sr)
    hop_size = int(0.100 * sr)

    if n_samples < block_size:
        # Fallback for very short snippets
        power = np.mean(y**2, axis=0)
        z = np.sum(power)
        if z < 1e-12:
            return -70.0
        return round(float(-0.691 + 10 * np.log10(z + 1e-12)), 1)

    n_blocks = (n_samples - block_size) // hop_size + 1
    block_powers = []

    for i in range(n_blocks):
        start = i * hop_size
        block = y[start:start + block_size, :]
        power = np.mean(block**2, axis=0)
        z = np.sum(power)
        block_powers.append(z)

    block_powers = np.array(block_powers)
    block_loudness = -0.691 + 10 * np.log10(np.maximum(block_powers, 1e-12))

    # Absolute threshold gate at -70 LUFS
    idx_abs = block_loudness > -70.0
    if not np.any(idx_abs):
        return -70.0

    # Relative threshold gate at -10 dB relative to un-gated mean
    z_avg = np.mean(block_powers[idx_abs])
    gamma_r = -0.691 + 10 * np.log10(z_avg + 1e-12) - 10.0

    idx_rel = block_loudness > gamma_r
    if not np.any(idx_rel):
        return round(float(gamma_r), 1)

    integrated_lufs = -0.691 + 10 * np.log10(np.mean(block_powers[idx_rel]) + 1e-12)
    return round(float(integrated_lufs), 1)

# ==========================================================
# 2. TRUE-PEAK & CREST FACTOR
# ==========================================================

def calculate_true_peak(audio: np.ndarray, sr: int = 44100, oversampling: int = 4) -> float:
    """
    Evaluates True-Peak in dBFS adhering to ITU-R BS.1770-4 using polyphase oversampling.
    Optimized: scans the highest peaks and resamples a localized window around the peak.
    """
    if audio is None or len(audio) == 0:
        return -70.0

    sample_peak = float(np.max(np.abs(audio)))
    if sample_peak < 1e-7:
        return -70.0

    # Locate peak sample across all channels
    mono = np.max(np.abs(audio), axis=1) if audio.ndim == 2 else np.abs(audio)
    top_idx = int(np.argmax(mono))

    # 1.5-second localized window around peak
    window = int(sr * 1.5)
    start_idx = max(0, top_idx - window)
    end_idx = min(len(audio), top_idx + window)
    slice_audio = audio[start_idx:end_idx].astype(np.float64)

    if len(slice_audio) > 0:
        # Reflect-pad boundaries to eliminate FIR step-transient overshoot
        pad_len = min(120, len(slice_audio) - 1)
        if pad_len > 1:
            if slice_audio.ndim == 2:
                padded = np.pad(slice_audio, ((pad_len, pad_len), (0, 0)), mode='reflect')
            else:
                padded = np.pad(slice_audio, (pad_len, pad_len), mode='reflect')
            audio_oversampled = signal.resample_poly(padded, oversampling, 1, axis=0)
            valid_oversampled = audio_oversampled[pad_len * oversampling : -pad_len * oversampling]
            tp_lin = max(sample_peak, float(np.max(np.abs(valid_oversampled))))
        else:
            audio_oversampled = signal.resample_poly(slice_audio, oversampling, 1, axis=0)
            tp_lin = max(sample_peak, float(np.max(np.abs(audio_oversampled))))
    else:
        tp_lin = sample_peak

    return round(float(20 * np.log10(tp_lin + 1e-12)), 2)

def calculate_crest_factor(audio: np.ndarray) -> float:
    """
    Measures dynamic range / crest factor in dB: 20 * log10(Peak / RMS).
    Higher crest factor indicates punchy transients; lower indicates compressed audio.
    """
    if audio is None or len(audio) == 0:
        return 0.0

    peak = float(np.max(np.abs(audio))) + 1e-9
    rms = float(np.sqrt(np.mean(audio**2))) + 1e-9
    return round(float(20 * np.log10(peak / rms)), 1)

# ==========================================================
# 3. SPECTRAL BALANCE & BANDS ANALYSIS
# ==========================================================

SPECTRAL_BANDS = {
    "sub": (20, 60),          # Sub-bass punch (808, kick sub)
    "low": (60, 250),         # Warmth & body (bass, kick body)
    "low_mid": (250, 800),     # Mud & boxiness (clutter zone)
    "mid": (800, 3500),       # Intelligibility & vocal presence
    "high_mid": (3500, 9000),  # Clarity, crispness & harshness
    "air": (9000, 22000)       # Studio sheen & air
}

def calculate_spectral_balance(audio: np.ndarray, sr: int = 44100) -> Dict[str, Any]:
    """
    Analyzes frequency energy distribution across critical production bands.
    Computes RMS power per band, percentage share, spectral centroid, and roll-off.
    """
    mono = np.mean(audio, axis=1) if (audio.ndim == 2 and audio.shape[1] > 1) else audio.squeeze()
    if len(mono) < 1024 or np.max(np.abs(mono)) < 1e-6:
        return {
            "bands_db": {k: -70.0 for k in SPECTRAL_BANDS},
            "bands_pct": {k: 0.0 for k in SPECTRAL_BANDS},
            "spectral_centroid_hz": 0,
            "rolloff_hz": 0,
            "is_dark": False,
            "is_bright": False,
            "has_excess_mud": False
        }

    nperseg = min(4096, len(mono))
    freqs, psd = signal.welch(mono, fs=sr, nperseg=nperseg)
    total_power = np.sum(psd) + 1e-12

    bands_db = {}
    bands_pct = {}

    for band_name, (f_min, f_max) in SPECTRAL_BANDS.items():
        mask = (freqs >= f_min) & (freqs < f_max)
        band_power = np.sum(psd[mask]) if np.any(mask) else 1e-12
        pct = (band_power / total_power) * 100.0
        db = 10 * np.log10(max(band_power, 1e-12))
        bands_db[band_name] = round(float(db), 1)
        bands_pct[band_name] = round(float(pct), 1)

    # Spectral Centroid (Center of Mass of frequencies)
    centroid = np.sum(freqs * psd) / total_power

    # Spectral Roll-off (frequency below which 85% of power resides)
    cumulative_power = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumulative_power, 0.85 * total_power)
    rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Heuristic classifications
    is_dark = centroid < 1200
    is_bright = centroid > 4500
    has_excess_mud = bands_pct.get("low_mid", 0) > 35.0

    return {
        "bands_db": bands_db,
        "bands_pct": bands_pct,
        "spectral_centroid_hz": round(float(centroid)),
        "rolloff_hz": round(rolloff_hz),
        "is_dark": bool(is_dark),
        "is_bright": bool(is_bright),
        "has_excess_mud": bool(has_excess_mud)
    }

# ==========================================================
# 4. SURGICAL RESONANCE & HARSHNESS DETECTION
# ==========================================================

RESONANCE_REGIONS = [
    {"name": "sub_boom", "range": (35, 95), "threshold": 4.5, "q": 4.0, "max_cut": 4.0},
    {"name": "mud_box", "range": (220, 520), "threshold": 3.8, "q": 3.5, "max_cut": 5.0},
    {"name": "nasal_honk", "range": (700, 1600), "threshold": 3.6, "q": 4.0, "max_cut": 4.5},
    {"name": "piercing_harsh", "range": (2400, 4800), "threshold": 3.5, "q": 4.5, "max_cut": 6.0},
    {"name": "sibilance_edge", "range": (5500, 9500), "threshold": 4.0, "q": 4.0, "max_cut": 4.5}
]

def detect_resonances(
    audio: np.ndarray,
    sr: int = 44100,
    sensitivity: float = 1.0,
    max_peaks: int = 6
) -> List[Dict[str, Any]]:
    """
    Detects prominent, narrow resonant spikes using Welch PSD and median filter baseline subtraction.
    Returns list of problem frequencies with estimated Q, prominence, and recommended cut.
    """
    mono = np.mean(audio, axis=1) if (audio.ndim == 2 and audio.shape[1] > 1) else audio.squeeze()
    if len(mono) < 2048 or np.max(np.abs(mono)) < 1e-4:
        return []

    nperseg = min(4096, len(mono))
    freqs, psd = signal.welch(mono, fs=sr, nperseg=nperseg)

    # Median filtering to produce smooth spectral baseline
    kernel = 41
    if len(psd) <= kernel:
        return []
    smooth_psd = signal.medfilt(psd, kernel_size=kernel)

    # Prominence in dB over baseline
    diff_db = 10.0 * np.log10(np.maximum(psd, 1e-12) / np.maximum(smooth_psd, 1e-12))

    detected = []
    sens = max(0.5, min(2.0, float(sensitivity)))

    for region in RESONANCE_REGIONS:
        f_low, f_high = region["range"]
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(mask):
            continue

        band_diff = diff_db[mask]
        band_f = freqs[mask]

        if len(band_diff) < 3:
            continue

        # Find local peaks inside this zone
        peaks, props = signal.find_peaks(band_diff, height=region["threshold"] / sens, distance=5)
        for p in peaks:
            peak_f = float(band_f[p])
            prominence = float(band_diff[p])
            
            # Attenuation proportional to prominence
            cut = min(region["max_cut"], prominence * 0.75)
            if cut >= 1.0:
                detected.append({
                    "region": region["name"],
                    "freq_hz": round(peak_f),
                    "prominence_db": round(prominence, 1),
                    "recommended_cut_db": -round(float(cut), 1),
                    "q": region["q"]
                })

    # Sort by prominence (highest problem peaks first) and cap to max_peaks
    detected.sort(key=lambda x: x["prominence_db"], reverse=True)
    return detected[:max_peaks]

# ==========================================================
# 5. ADAPTIVE STEM CHARACTERIZATION & TUNING
# ==========================================================

TARGET_STEM_LOUDNESS_LUFS = {
    "kick": -15.0,
    "bass": -17.0,
    "vocal_lead": -19.0,
    "vocal_back": -23.0,
    "snare_clap": -17.5,
    "percussion_high": -22.0,
    "acoustic_strum": -21.0,
    "harp_keys": -22.0,
    "strings": -23.0,
    "other": -21.5
}

def analyze_stem(
    audio: np.ndarray,
    sr: int,
    stem_type: str = "other",
    stem_name: str = ""
) -> Dict[str, Any]:
    """
    Performs comprehensive acoustic analysis on a single stem.
    Calculates LUFS, true peak, spectral distribution, detected resonances,
    and adaptive parameter recommendations (gain offset, HPF cutoff, resonance notch).
    """
    lufs = calculate_lufs(audio, sr)
    tp = calculate_true_peak(audio, sr)
    crest = calculate_crest_factor(audio)
    spectral = calculate_spectral_balance(audio, sr)
    resonances = detect_resonances(audio, sr, sensitivity=1.0, max_peaks=4)

    # 1. Adaptive Gain Offset (normalizing stem foundation before genre coloring)
    target_lufs = TARGET_STEM_LOUDNESS_LUFS.get(stem_type, -20.0)
    if lufs > -60.0:
        raw_gain_delta = target_lufs - lufs
        # Clamp adaptive trim between -4.0 dB and +4.0 dB to prevent extreme shifts
        adaptive_gain_db = max(-4.0, min(4.0, raw_gain_delta))
    else:
        adaptive_gain_db = 0.0

    # 2. Adaptive High-Pass Filter (HPF)
    # Detect where significant low energy begins to avoid cutting body while clearing sub-rumble
    bands = spectral["bands_pct"]
    sub_pct = bands.get("sub", 0.0)
    low_pct = bands.get("low", 0.0)

    adaptive_hpf_hz = None
    if stem_type in ["vocal_lead", "vocal_back"]:
        # If vocal has high sub energy (>1.5%), there is mic rumble / proximity effect -> HPF at 115 Hz
        if sub_pct > 1.5:
            adaptive_hpf_hz = 115
        elif sub_pct > 0.5:
            adaptive_hpf_hz = 95
        else:
            adaptive_hpf_hz = 80
    elif stem_type == "bass":
        # If 808 with high sub, protect 26Hz; if standard bass, 32Hz
        adaptive_hpf_hz = 26 if sub_pct > 18.0 else 32
    elif stem_type in ["acoustic_strum", "harp_keys", "strings"]:
        adaptive_hpf_hz = 120 if sub_pct > 2.0 else 90

    return {
        "stem_name": stem_name,
        "stem_type": stem_type,
        "lufs": lufs,
        "true_peak_dbfs": tp,
        "crest_factor_db": crest,
        "spectral_balance": spectral,
        "resonances": resonances,
        "recommendations": {
            "adaptive_gain_db": round(float(adaptive_gain_db), 1),
            "adaptive_hpf_hz": adaptive_hpf_hz,
            "tame_resonances": resonances
        }
    }
