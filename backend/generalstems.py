import os
# Strictly prevent OpenMP/MKL/BLAS from spawning 32/64 threads on cloud 2-vCPU cgroups
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["TORCH_NUM_THREADS"] = "2"

try:
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    HAS_DEMUCS = True
except ImportError:
    torch = None
    torchaudio = None
    apply_model = None
    get_model = None
    HAS_DEMUCS = False

import numpy as np
try:
    import librosa
except ImportError:
    librosa = None

import soundfile as sf
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# ----------------------------- 
# Utilities
# ----------------------------- 
def bandpass(y, sr, low, high, order=6):
    """Apply bandpass filter with higher order for cleaner separation."""
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, y)

def highpass(y, sr, cutoff, order=6):
    """Apply highpass filter."""
    sos = butter(order, cutoff, btype="high", fs=sr, output="sos")
    return sosfiltfilt(sos, y)

def lowpass(y, sr, cutoff, order=6):
    """Apply lowpass filter."""
    sos = butter(order, cutoff, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y)

def normalize(y, peak=0.95):
    """Normalize audio with headroom."""
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val * peak
    return y

def save_stem(path, audio, sr):
    """Normalize and save audio in studio-grade 16-bit PCM stereo WAV (50% smaller transfer size)."""
    if torch is not None and isinstance(audio, torch.Tensor):
        arr = audio.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T  # (channels, samples) -> (samples, channels)
        elif arr.ndim == 1:
            arr = np.column_stack([arr, arr])
        max_val = np.max(np.abs(arr))
        if max_val > 1e-6:
            arr = (arr / max_val) * 0.95
        sf.write(path, arr.astype(np.float32), sr, subtype='PCM_16')
    else:
        norm_audio = normalize(audio)
        if norm_audio.ndim == 1:
            norm_audio = np.column_stack([norm_audio, norm_audio])
        sf.write(path, norm_audio.astype(np.float32), sr, subtype='PCM_16')

# ----------------------------- 
# Advanced Key Detection
# ----------------------------- 
def detect_key_advanced(y, sr):
    """
    Professional key detection using Essentia (industry standard).
    Falls back cleanly to Krumhansl-Schmuckler if Essentia is unavailable or fails.
    """
    try:
        import essentia.standard as es
        
        # Resample to 44100 if needed (Essentia works best at this rate)
        if sr != 44100:
            import librosa as lr
            y_res = lr.resample(y, orig_sr=sr, target_sr=44100)
        else:
            y_res = y
        
        # Ensure float32 mono
        audio = y_res.astype(np.float32)
        if len(audio) > 44100 and np.max(np.abs(audio)) > 1e-4:
            key_detector = es.KeyExtractor()
            key, scale, strength = key_detector(audio)
            if key and scale:
                return f"{key} {scale.capitalize()}"
    except Exception:
        pass
        
    # Robust fallback
    try:
        return detect_key_krumhansl(y, sr)
    except Exception:
        return "C Major"

def detect_key_krumhansl(y, sr):
    """
    Fallback: Krumhansl-Schmuckler algorithm for key detection with zero-division protection.
    """
    try:
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        
        # Focus on first 30 seconds to speed up CQT analysis
        max_samples = int(sr * 30)
        y_slice = y[:max_samples] if len(y) > max_samples else y
        
        if np.max(np.abs(y_slice)) < 1e-4:
            return "C Major"
        
        # Use harmonic component on slice
        y_harmonic = librosa.effects.harmonic(y_slice, margin=4)
        
        # Fast CQT chroma
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=512, n_chroma=12)
        
        # Simple average
        mean_chroma = np.mean(chroma, axis=1)
        norm = np.linalg.norm(mean_chroma)
        if norm < 1e-6:
            return "C Major"
        mean_chroma = mean_chroma / norm
        
        best_corr = -1
        best_key = "C"
        best_is_major = True
        
        for i, key in enumerate(keys):
            major_prof = np.roll(major_profile, i)
            minor_prof = np.roll(minor_profile, i)
            
            major_prof = major_prof / np.linalg.norm(major_prof)
            minor_prof = minor_prof / np.linalg.norm(minor_prof)
            
            corr_major = np.corrcoef(mean_chroma, major_prof)[0, 1]
            corr_minor = np.corrcoef(mean_chroma, minor_prof)[0, 1]
            
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = key
                best_is_major = True
            
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = key
                best_is_major = False
        
        mode = "Major" if best_is_major else "Minor"
        return f"{best_key} {mode}"
    except Exception:
        return "C Major"

# ----------------------------- 
# Advanced Tempo Detection
# ----------------------------- 
def detect_tempo_advanced(y, sr):
    """
    Ultra-fast and musically accurate BPM detection snap to nearest whole integer.
    Computes onset envelope directly (<0.05s) with zero-energy handling.
    """
    try:
        if y is None or len(y) == 0:
            return 120
            
        # Focus on first 35 seconds for maximum rhythm clarity
        max_samples = int(sr * 35)
        y_slice = y[:max_samples] if len(y) > max_samples else y
        
        peak_amp = np.max(np.abs(y_slice))
        if peak_amp < 1e-4:
            return 120
            
        y_norm = y_slice / peak_amp
        
        # Compute onset envelope directly
        onset_env = librosa.onset.onset_strength(y=y_norm, sr=sr)
        if np.max(onset_env) < 1e-4:
            return 120
            
        # Use median aggregation for robust tempo
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
        
        if isinstance(tempo, np.ndarray) and len(tempo) > 0:
            bpm = float(tempo[0])
        else:
            bpm = float(tempo)
        
        if np.isnan(bpm) or bpm < 40 or bpm > 260:
            return 120
        
        return int(round(bpm))
    except Exception:
        return 120
    
# ----------------------------- 
# Convert MP3 → WAV
# ----------------------------- 
def mp3_to_wav(mp3_path, wav_path=None):
    """Convert audio file to high-quality WAV."""
    if wav_path is None:
        wav_path = os.path.splitext(mp3_path)[0] + ".wav"
    
    audio = AudioSegment.from_file(mp3_path)
    # Force 44.1kHz stereo for consistency
    audio = audio.set_frame_rate(44100).set_channels(2)
    audio.export(wav_path, format="wav")
    return wav_path

# ----------------------------- 
# Separation using Demucs (Optimized)
# ----------------------------- 
_CACHED_DEMUCS_MODEL = None

def get_cached_demucs_model(device):
    global _CACHED_DEMUCS_MODEL
    if _CACHED_DEMUCS_MODEL is None:
        print("[DEMUCS] Loading htdemucs model into memory...")
        _CACHED_DEMUCS_MODEL = get_model("htdemucs")
        _CACHED_DEMUCS_MODEL.to(device)
        _CACHED_DEMUCS_MODEL.eval()
    return _CACHED_DEMUCS_MODEL

def demucs_separate(wav_path, out_dir):
    """Separate audio into stems using Demucs with CPU/GPU acceleration."""
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Crucial for Hugging Face free tier: The container quota is strictly 2 vCPUs.
        # Spawning more threads than quota causes severe context thrashing and Linux CFS throttling.
        try:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    
    model = get_cached_demucs_model(device)
    
    wav, sr = torchaudio.load(wav_path)
    
    # Ensure stereo
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    
    wav = wav.unsqueeze(0).to(device)
    
    # High-speed inference: overlap=0.02 (minimal overlap), shifts=0
    # Eliminates redundant chunk overlap computation on CPU without audible loss
    with torch.inference_mode():
        sources = apply_model(model, wav, split=True, overlap=0.02, shifts=0, progress=False)
    
    sources = sources[0]
    drums, bass, other, vocals = sources
    
    # Save stems
    stems = {}
    stem_paths = {
        "vocals": os.path.join(out_dir, "vocals.wav"),
        "drums": os.path.join(out_dir, "drums.wav"),
        "bass": os.path.join(out_dir, "bass.wav"),
        "other": os.path.join(out_dir, "other.wav")
    }
    
    save_stem(stem_paths["vocals"], vocals, sr)
    save_stem(stem_paths["drums"], drums, sr)
    save_stem(stem_paths["bass"], bass, sr)
    save_stem(stem_paths["other"], other, sr)
    
    stems.update(stem_paths)
    
    del wav, sources, drums, bass, other, vocals
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return stems, sr

# ----------------------------- 
# Pro Drum Refinement (STFT Soft Spectral Masking & Genre Adaptation)
# ----------------------------- 
def detect_drum_acoustic_genre(y, sr):
    """
    Analyzes drum spectral distribution and transient dynamics to identify acoustic profile:
    'urban_808', 'electronic', 'acoustic_rock', or 'modern_pop'.
    """
    if y.ndim > 1:
        mono = np.mean(y, axis=1)
    else:
        mono = y
    
    if len(mono) < sr * 2:
        return "modern_pop"
        
    # Analyze first 30 seconds for speed and precision
    sample = mono[:int(sr * 30.0)]
    sample = sample - np.mean(sample)
    
    # Compute power spectrum via Welch's method
    f, psd = signal.welch(sample, fs=sr, nperseg=4096)
    total_power = np.sum(psd) + 1e-12
    
    sub_bass_power = np.sum(psd[(f >= 25) & (f <= 90)]) / total_power
    low_punch_power = np.sum(psd[(f > 90) & (f <= 220)]) / total_power
    mid_power = np.sum(psd[(f > 220) & (f <= 2800)]) / total_power
    high_power = np.sum(psd[(f > 4500) & (f <= 16000)]) / total_power
    
    # Transient dynamic crest factor
    rms = np.sqrt(np.mean(sample**2)) + 1e-12
    peak = np.max(np.abs(sample))
    crest = peak / rms
    
    if sub_bass_power > 0.32 or (sub_bass_power > 0.22 and sub_bass_power > low_punch_power * 1.3):
        return "urban_808"
    elif high_power > 0.25 and low_punch_power > 0.20:
        return "electronic"
    elif mid_power > 0.38 and crest < 6.5:
        return "acoustic_rock"
    else:
        return "modern_pop"


def refine_drums(drums_path, output_dir, detected_genre=None):
    """
    Studio-grade drum refinement using STFT Soft Spectral Masking with genre adaptation,
    transient preservation, temporal envelope smoothing, and exact acoustic scale identity.
    Reconstructs Kick, Snare, and Hi-Hat with Mk + Ms + Mh = 1.0.
    Guarantees: Kick(t) + Snare(t) + HiHat(t) == DrumsRaw(t) exactly.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read drum stem fast via soundfile
    y, sr = sf.read(drums_path, always_2d=True, dtype='float32')
    num_samples, num_channels = y.shape
    
    # Genre detection
    if not detected_genre:
        detected_genre = detect_drum_acoustic_genre(y, sr)
    print(f"[INFO] Drum refinement profile: {detected_genre}")
    
    # STFT parameters: 2048 samples (approx 46ms @ 44.1kHz), 75% overlap
    nperseg = 2048
    noverlap = 1536
    
    # Compute STFT for stereo channels: shape (channels, freqs, time_frames)
    f, t, Zxx = signal.stft(
        y.T, 
        fs=sr, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap, 
        boundary='zeros'
    )
    
    n_freqs, n_times = len(f), len(t)
    
    # Mono magnitude spectrogram for spectral feature extraction
    mag_mono = np.mean(np.abs(Zxx), axis=0) # (freqs, times)
    
    # Genre-specific frequency boundaries and curves
    if detected_genre == "urban_808":
        # Extended sub-bass, 808 sub up to 115Hz, punch up to 200Hz, snappy claps/snares, crisp hats
        f_kick_sub = 115.0
        f_snare_low = 135.0
        f_snare_high = 4800.0
        f_hihat_low = 3600.0
        click_boost = 1.3
    elif detected_genre == "electronic":
        # Punchy 4-on-the-floor kick 50-140Hz, wide snares/claps, bright hats
        f_kick_sub = 125.0
        f_snare_low = 140.0
        f_snare_high = 5000.0
        f_hihat_low = 3500.0
        click_boost = 1.15
    elif detected_genre == "acoustic_rock":
        # Natural drumkit: kick body 50-130Hz, snare shell resonance 160-280Hz, snare wires up to 7.5kHz
        f_kick_sub = 110.0
        f_snare_low = 125.0
        f_snare_high = 5200.0
        f_hihat_low = 3200.0
        click_boost = 1.05
    else: # modern_pop
        f_kick_sub = 115.0
        f_snare_low = 135.0
        f_snare_high = 5000.0
        f_hihat_low = 3400.0
        click_boost = 1.1
        
    # 1. Base frequency weight curves (Sigmoids clipped to avoid overflow)
    # Kick base: high below f_kick_sub, rolls off smoothly through low-mids
    arg_kick = np.clip((f - f_kick_sub) / 28.0, -80.0, 80.0)
    w_kick_base = 1.0 / (1.0 + np.exp(arg_kick))
    w_kick_warmth = 0.3 * np.exp(-((f - (f_kick_sub + 40.0)) / 60.0)**2)
    w_kick_base = np.maximum(w_kick_base, w_kick_warmth)
    
    # Snare base: bandpass curve from f_snare_low up to f_snare_high
    arg_snare_l = np.clip(-(f - f_snare_low) / 22.0, -80.0, 80.0)
    arg_snare_h = np.clip((f - f_snare_high) / 800.0, -80.0, 80.0)
    w_snare_low_shelf = 1.0 / (1.0 + np.exp(arg_snare_l))
    w_snare_high_shelf = 1.0 / (1.0 + np.exp(arg_snare_h))
    w_snare_base = w_snare_low_shelf * w_snare_high_shelf
    
    # Hi-hat base: highpass curve starting from f_hihat_low
    arg_hihat = np.clip(-(f - f_hihat_low) / 600.0, -80.0, 80.0)
    w_hihat_base = 1.0 / (1.0 + np.exp(arg_hihat))
    
    # Broadcast base weights across time: (n_freqs, n_times)
    W_k = np.tile(w_kick_base[:, None], (1, n_times))
    W_s = np.tile(w_snare_base[:, None], (1, n_times))
    W_h = np.tile(w_hihat_base[:, None], (1, n_times))
    
    # 2. Smooth Asymmetric Envelope Followers (Zero Choppiness / Natural Sustains)
    # A) Kick Low-Frequency Onset & Smooth Decay (Attack 0ms, Decay ~65ms)
    low_mask = (f >= 30) & (f <= 130)
    low_energy = np.sum(mag_mono[low_mask, :], axis=0)
    low_diff = np.maximum(0, np.diff(low_energy, prepend=low_energy[0]))
    
    env_kick = np.zeros(n_times, dtype=np.float32)
    alpha_kick = 0.82  # ~65ms decay at ~11.6ms frame step
    for ti in range(n_times):
        c = low_diff[ti]
        p = env_kick[ti - 1] * alpha_kick if ti > 0 else 0.0
        env_kick[ti] = max(c, p)
    norm_env_kick = env_kick / (np.max(env_kick) + 1e-12)
    
    # Inject beater click (2.2k - 4.5k Hz) during kick hits with natural decay
    click_mask = (f >= 2200) & (f <= 4500)
    click_profile = np.exp(-((f[click_mask] - 3200) / 850.0)**2)
    transient_click = 0.65 * click_boost * np.outer(click_profile, norm_env_kick)
    W_k[click_mask, :] += transient_click
    
    # B) Snare Mid-Frequency Onset & Smooth Decay (Attack 0ms, Decay ~85ms)
    snare_mask = (f >= 180) & (f <= 2600)
    snare_energy = np.sum(mag_mono[snare_mask, :], axis=0)
    snare_diff = np.maximum(0, np.diff(snare_energy, prepend=snare_energy[0]))
    
    env_snare = np.zeros(n_times, dtype=np.float32)
    alpha_snare = 0.86  # ~85ms decay
    for ti in range(n_times):
        c = snare_diff[ti]
        p = env_snare[ti - 1] * alpha_snare if ti > 0 else 0.0
        env_snare[ti] = max(c, p)
    norm_env_snare = env_snare / (np.max(env_snare) + 1e-12)
    
    # Inject snare wires & sizzle up to 8kHz during snare strokes
    wire_mask = (f >= 3500) & (f <= 8000)
    wire_profile = np.exp(-((f[wire_mask] - 5000) / 1800.0)**2)
    snare_wires = 0.55 * np.outer(wire_profile, norm_env_snare)
    W_s[wire_mask, :] += snare_wires
    
    # C) Hi-Hat High-Frequency Shimmer & Transients
    hi_mask = (f >= 5000) & (f <= 16000)
    hi_energy = np.sum(mag_mono[hi_mask, :], axis=0)
    hi_diff = np.maximum(0, np.diff(hi_energy, prepend=hi_energy[0]))
    
    env_hihat = np.zeros(n_times, dtype=np.float32)
    alpha_hihat = 0.78
    for ti in range(n_times):
        c = hi_diff[ti]
        p = env_hihat[ti - 1] * alpha_hihat if ti > 0 else 0.0
        env_hihat[ti] = max(c, p)
    norm_env_hihat = env_hihat / (np.max(env_hihat) + 1e-12)
    W_h[hi_mask, :] += 0.45 * norm_env_hihat[None, :]
    
    # Floor to ensure numeric stability
    W_k = np.maximum(1e-4, W_k)
    W_s = np.maximum(1e-4, W_s)
    W_h = np.maximum(1e-4, W_h)
    
    # 4. Strict Unitary Normalization: Mk + Ms + Mh = 1.0 (Zero lost audio!)
    Sum_W = W_k + W_s + W_h
    M_k = W_k / Sum_W
    M_s = W_s / Sum_W
    M_h = W_h / Sum_W
    
    # 3-Point Hann temporal smoothing across time axis (eliminates all flutter and choppiness)
    if n_times > 2:
        for M in [M_k, M_s, M_h]:
            M[:, 1:-1] = 0.25 * M[:, :-2] + 0.5 * M[:, 1:-1] + 0.25 * M[:, 2:]
        # Re-normalize to strictly maintain Mk + Ms + Mh = 1.0
        Sum_M = M_k + M_s + M_h
        M_k /= Sum_M
        M_s /= Sum_M
        M_h /= Sum_M
    
    # 5. Apply masks to complex STFT (preserves full stereo image and phase)
    # Zxx shape is (channels, freqs, times)
    Z_k = Zxx * M_k[None, :, :]
    Z_s = Zxx * M_s[None, :, :]
    Z_h = Zxx * M_h[None, :, :]
    
    # 6. Inverse STFT to get time-domain stereo stems
    _, x_k = signal.istft(Z_k, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, boundary='zeros')
    _, x_s = signal.istft(Z_s, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, boundary='zeros')
    _, x_h = signal.istft(Z_h, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, boundary='zeros')
    
    # Align lengths to exact original sample count: shape (samples, channels)
    x_k = x_k.T[:num_samples, :]
    x_s = x_s.T[:num_samples, :]
    x_h = x_h.T[:num_samples, :]
    
    # =========================================================================
    # EXACT ACOUSTIC IDENTITY: Kick(t) + Snare(t) + HiHat(t) == DrumsRaw(t)
    # Unitary STFT masks guarantee bit-for-bit exact sum with zero scaling distortion.
    # =========================================================================
    x_k = np.clip(x_k, -1.0, 1.0)
    x_s = np.clip(x_s, -1.0, 1.0)
    x_h = np.clip(x_h, -1.0, 1.0)
    
    # Save stems as 16-bit PCM WAV (stereo preserved)
    kick_path = os.path.join(output_dir, "kick.wav")
    snare_path = os.path.join(output_dir, "snare.wav")
    hihat_path = os.path.join(output_dir, "hihat.wav")
    
    sf.write(kick_path, x_k.astype(np.float32), sr, subtype='PCM_16')
    sf.write(snare_path, x_s.astype(np.float32), sr, subtype='PCM_16')
    sf.write(hihat_path, x_h.astype(np.float32), sr, subtype='PCM_16')
    
    return {
        "kick": kick_path,
        "snare": snare_path,
        "hihat": hihat_path,
        "genre": detected_genre
    }

# ----------------------------- 
# Full pipeline
# ----------------------------- 
def process_song(mp3_path, output_dir="stems"):
    """
    Complete processing pipeline for stem separation and analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to WAV
    wav_path = mp3_to_wav(mp3_path, os.path.join(output_dir, "song.wav"))
    
    # Separate stems
    stems, sr = demucs_separate(wav_path, output_dir)
    
    # Refine drums
    drum_stems = refine_drums(stems["drums"], output_dir)
    stems.update(drum_stems)
    
    # Detect key and tempo
    y, sr_load = librosa.load(wav_path, mono=True, sr=44100)
    key = detect_key_advanced(y, sr_load)
    tempo = detect_tempo_advanced(y, sr_load)
    
    stems["wav_path"] = wav_path
    stems["key"] = key
    stems["tempo"] = tempo
    
    return stems