import os
# Strictly prevent OpenMP/MKL/BLAS from spawning 32/64 threads on cloud 2-vCPU cgroups
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["TORCH_NUM_THREADS"] = "2"

import torch
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import torchaudio
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter
from pydub import AudioSegment
from demucs.apply import apply_model
from demucs.pretrained import get_model

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
    """Normalize and save audio in studio-grade 32-bit Float WAV."""
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T  # (channels, samples) -> (samples, channels)
        max_val = np.max(np.abs(arr))
        if max_val > 1e-6:
            arr = (arr / max_val) * 0.95
        sf.write(path, arr.astype(np.float32), sr, subtype='FLOAT')
    else:
        norm_audio = normalize(audio)
        sf.write(path, norm_audio.astype(np.float32), sr, subtype='FLOAT')

# ----------------------------- 
# Advanced Key Detection
# ----------------------------- 
def detect_key_advanced(y, sr):
    """
    Professional key detection using Essentia (industry standard).
    Falls back to Krumhansl-Schmuckler if Essentia is unavailable.
    """
    try:
        import essentia.standard as es
        
        # Resample to 44100 if needed (Essentia works best at this rate)
        if sr != 44100:
            import librosa as lr
            y = lr.resample(y, orig_sr=sr, target_sr=44100)
            sr = 44100
        
        # Ensure float32
        audio = y.astype(np.float32)
        
        # Use Essentia's Key detector (used by Spotify, Beatport, etc.)
        key_detector = es.KeyExtractor()
        key, scale, strength = key_detector(audio)
        
        # Essentia returns scale as "major" or "minor"
        return f"{key} {scale}"
        
    except ImportError:
        # Fallback to Krumhansl-Schmuckler if Essentia not available
        return detect_key_krumhansl(y, sr)

def detect_key_krumhansl(y, sr):
    """
    Fallback: Krumhansl-Schmuckler algorithm for key detection.
    """
    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    # Focus on first 30 seconds to speed up CQT analysis by 10x
    max_samples = int(sr * 30)
    y_slice = y[:max_samples] if len(y) > max_samples else y
    
    # Use harmonic component on slice
    y_harmonic = librosa.effects.harmonic(y_slice, margin=4)
    
    # Fast CQT chroma
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=512, n_chroma=12)
    
    # Simple average
    mean_chroma = np.mean(chroma, axis=1)
    mean_chroma = mean_chroma / np.linalg.norm(mean_chroma)
    
    best_corr = -1
    best_key = None
    best_is_major = True
    
    for i, key in enumerate(keys):
        # Roll profiles
        major_prof = np.roll(major_profile, i)
        minor_prof = np.roll(minor_profile, i)
        
        # Normalize
        major_prof = major_prof / np.linalg.norm(major_prof)
        minor_prof = minor_prof / np.linalg.norm(minor_prof)
        
        # Calculate correlation
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
    
    # Format output
    mode = "major" if best_is_major else "minor"
    return f"{best_key} {mode}"

# ----------------------------- 
# Advanced Tempo Detection
# ----------------------------- 
def detect_tempo_advanced(y, sr):
    """
    Ultra-fast and musically accurate BPM detection snap to nearest whole integer.
    Computes onset envelope directly (<0.05s) avoiding heavy STFT harmonic-percussive separation.
    """
    # Focus on first 30 seconds for maximum rhythm clarity and 10x faster execution
    max_samples = int(sr * 30)
    y_slice = y[:max_samples] if len(y) > max_samples else y
    
    # Compute onset envelope directly
    onset_env = librosa.onset.onset_strength(y=y_slice, sr=sr)
    
    # Use median aggregation for robust tempo
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
    
    # Extract the value (librosa returns ndarray)
    if isinstance(tempo, np.ndarray):
        bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        bpm = float(tempo)
    
    return int(round(bpm))
    
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
# Enhanced Drum Refinement (High-Speed DSP)
# ----------------------------- 
def refine_drums(drums_path, output_dir):
    """
    Refine drum stems into kick, snare, and hi-hat with high-speed zero-phase digital filtering.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read drum stem fast via soundfile
    y, sr = sf.read(drums_path, always_2d=True, dtype='float32')
    y_mono = np.mean(y, axis=1)
    y_mono = y_mono - np.mean(y_mono)
    
    # Direct percussive content from isolated Demucs drums
    perc = y_mono
    
    # ========== KICK DRUM ==========
    # Kick: 40-150 Hz
    kick_raw = bandpass(perc, sr, 40, 150, order=4)
    kick = normalize(kick_raw)
    kick_path = os.path.join(output_dir, "kick.wav")
    save_stem(kick_path, kick, sr)
    
    # ========== SNARE DRUM ==========
    # Snare: 180-3000 Hz
    snare_raw = bandpass(perc, sr, 180, 3000, order=4)
    kick_bleed = lowpass(kick_raw, sr, 200, order=4)
    snare_clean = snare_raw - (kick_bleed * 0.7)
    snare_clean = normalize(snare_clean)
    snare_path = os.path.join(output_dir, "snare.wav")
    save_stem(snare_path, snare_clean, sr)
    
    # ========== HI-HAT ==========
    # Hi-hat: 6000-20000 Hz
    hihat_raw = highpass(perc, sr, 6000, order=4)
    kick_bleed_hi = lowpass(kick_raw, sr, 300, order=4)
    snare_bleed_hi = bandpass(snare_clean, sr, 200, 2000, order=4)
    hihat_clean = hihat_raw - (kick_bleed_hi * 0.2) - (snare_bleed_hi * 0.15)
    hihat_clean = normalize(hihat_clean)
    hihat_path = os.path.join(output_dir, "hihat.wav")
    save_stem(hihat_path, hihat_clean, sr)
    
    return {
        "kick": kick_path,
        "snare": snare_path,
        "hihat": hihat_path
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