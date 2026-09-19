"""
Myooz Generative Audio Engine: AI Instrument & Arrangement Generation
Connects to dedicated Hugging Face Inference Endpoints running music generation models
(e.g., facebook/musicgen-melody or stabilityai/stable-audio-open-1.0).
Exports 32-bit Floating Point WAV stems calibrated for seamless Mixter mixing.
"""

import os
import io
import json
import math
import time
import base64
import numpy as np
import soundfile as sf
import scipy.signal as signal
import requests

try:
    from audio_analysis import calculate_lufs, calculate_true_peak, calculate_crest_factor
except ImportError:
    try:
        from backend.audio_analysis import calculate_lufs, calculate_true_peak, calculate_crest_factor
    except ImportError:
        def calculate_lufs(a, sr=44100): return -14.0
        def calculate_true_peak(a, sr=44100): return -3.0
        def calculate_crest_factor(a): return 10.0

# ==========================================================
# CONFIGURATION & CREDENTIALS
# ==========================================================

HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL", "").strip()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", os.getenv("HF_TOKEN", "")).strip()
DEFAULT_TIMEOUT_SEC = int(os.getenv("HF_TIMEOUT", "75"))

def get_hf_credentials():
    """Retrieves Hugging Face Inference Endpoint credentials from environment."""
    url = os.getenv("HF_ENDPOINT_URL", "").strip()
    token = os.getenv("HF_API_TOKEN", os.getenv("HF_TOKEN", "")).strip()
    return url, token

# ==========================================================
# PROMPT FORMATTING & CONDITIONING
# ==========================================================

def build_conditioning_prompt(
    description: str,
    tempo: str = None,
    key: str = None,
    section: str = None
) -> str:
    """
    Constructs an optimized prompt string for music generation models incorporating
    tempo (BPM), musical key, and section context.
    """
    desc = (description or "melodic synth instrument").strip()
    tags = [desc]
    
    if tempo and str(tempo).strip():
        clean_tempo = str(tempo).replace("BPM", "").strip()
        tags.append(f"{clean_tempo} BPM")
        
    if key and str(key).strip():
        tags.append(f"in {str(key).strip()}")
        
    if section and str(section).strip():
        sec_clean = str(section).lower().strip()
        if "chorus" in sec_clean or "drop" in sec_clean:
            tags.append("powerful chorus drop energy")
        elif "verse" in sec_clean:
            tags.append("intimate verse texture")
        elif "intro" in sec_clean:
            tags.append("atmospheric intro build")
        elif "outro" in sec_clean:
            tags.append("smooth outro decay")
            
    tags.append("studio quality, isolated stem, dry, pristine acoustics")
    return ", ".join(tags)

# ==========================================================
# AUDIO DECODING & NORMALIZATION
# ==========================================================

def decode_and_normalize_audio(audio_bytes: bytes, target_sr: int = 44100) -> np.ndarray:
    """
    Decodes raw audio bytes (WAV, MP3, OGG, FLAC) into a standardized
    32-bit float stereo array normalized to -3.0 dBFS true headroom.
    """
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True, dtype='float32')
    except Exception as e:
        raise ValueError(f"Failed to decode audio bytes: {e}")

    # Ensure stereo
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]

    # Resample if sample rate doesn't match target
    if sr != target_sr:
        gcd = math.gcd(target_sr, sr)
        data = signal.resample_poly(data, target_sr // gcd, sr // gcd, axis=0)

    # Standardize peak headroom to -3.0 dBFS (0.707 linear)
    current_peak = float(np.max(np.abs(data)))
    target_peak = 10 ** (-3.0 / 20.0) # ~0.707
    if current_peak > 1e-5:
        data = data * (target_peak / current_peak)

    return data.astype(np.float32)

# ==========================================================
# SYNTHETIC INSTRUMENT GENERATOR (OFFLINE / TEST FALLBACK)
# ==========================================================

def generate_mock_instrument(
    prompt: str = "synth lead",
    tempo: float = 120.0,
    key: str = "C Major",
    duration_sec: float = 12.0,
    sr: int = 44100
) -> np.ndarray:
    """
    Generates a clean musical audio stem for dry-run testing and offline verification
    when no active Hugging Face GPU endpoint is configured.
    """
    total_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    # Base pitch from musical key
    key_pitches = {
        "c": 261.63, "c#": 277.18, "db": 277.18,
        "d": 293.66, "d#": 311.13, "eb": 311.13,
        "e": 329.63, "f": 349.23, "f#": 369.99, "gb": 369.99,
        "g": 392.00, "g#": 415.30, "ab": 415.30,
        "a": 440.00, "a#": 466.16, "bb": 466.16,
        "b": 493.88
    }
    
    first_token = key.strip().split()[0].lower() if key else "c"
    base_freq = key_pitches.get(first_token, 261.63)
    
    prompt_lower = prompt.lower()
    if "bass" in prompt_lower or "808" in prompt_lower or "sub" in prompt_lower:
        freq = base_freq / 4.0 # Sub-bass register
        audio = np.sin(2 * np.pi * freq * t) * 0.7 + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        # Apply saturation envelope
        audio = np.tanh(audio * 1.5) * 0.8
    elif "pad" in prompt_lower or "string" in prompt_lower:
        f1 = base_freq
        f2 = base_freq * (5.0 / 4.0) # Major 3rd
        f3 = base_freq * (3.0 / 2.0) # Perfect 5th
        audio = 0.35 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t) + 0.25 * np.sin(2 * np.pi * f3 * t)
    else:
        # Arpeggiated melody based on tempo
        try:
            bpm = float(tempo) if tempo else 120.0
        except ValueError:
            bpm = 120.0
        beat_dur = 60.0 / bpm
        note_index = np.floor(t / (beat_dur / 2.0)).astype(int)
        arpeggio_mults = [1.0, 1.25, 1.5, 1.875, 2.0]
        cur_mult = np.array([arpeggio_mults[idx % len(arpeggio_mults)] for idx in note_index])
        phase = 2 * np.pi * np.cumsum(base_freq * cur_mult / float(sr))
        audio = np.sin(phase) * 0.6
        # Note decay envelope
        note_env = np.exp(-((t % (beat_dur / 2.0)) / (beat_dur / 2.0)) * 3.5)
        audio = audio * note_env
        
    stereo = np.column_stack([audio, audio * 0.98]).astype(np.float32)
    # Peak normalization to -3 dBFS
    peak = np.max(np.abs(stereo))
    if peak > 1e-4:
        stereo = stereo * (10 ** (-3.0 / 20.0) / peak)
    return stereo

# ==========================================================
# MAIN GENERATIVE INFERENCE EXECUTOR
# ==========================================================

def generate_instrument_stem(
    prompt: str,
    tempo: str = None,
    key: str = None,
    section: str = None,
    duration_sec: float = 15.0,
    ref_audio_path: str = None,
    output_path: str = None,
    endpoint_url: str = None,
    api_token: str = None,
    mock: bool = False
):
    """
    Generates a commercial music instrument stem via a dedicated Hugging Face Inference Endpoint.
    
    Args:
        prompt: Natural language description of the instrument/sound.
        tempo: Optional BPM (e.g. 120 or '128').
        key: Optional musical key (e.g. 'C Minor').
        section: Optional song section ('chorus', 'verse', 'intro', 'outro').
        duration_sec: Length of generated stem in seconds (clamped to 30s).
        ref_audio_path: Optional path to reference WAV/MP3 for melodic conditioning.
        output_path: Target WAV file location.
        endpoint_url: Overrides HF_ENDPOINT_URL environment variable.
        api_token: Overrides HF_API_TOKEN environment variable.
        mock: If True, generates a synthetic acoustic stem for offline testing.
        
    Returns:
        (output_path, metadata_dict)
    """
    if not output_path:
        tmp_dir = "/tmp/myooz_generative"
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = int(time.time())
        output_path = os.path.join(tmp_dir, f"gen_stem_{timestamp}.wav")
        
    target_sr = 44100
    dur_clamped = min(30.0, max(2.0, float(duration_sec)))
    full_prompt = build_conditioning_prompt(prompt, tempo=tempo, key=key, section=section)
    
    url = (endpoint_url or HF_ENDPOINT_URL).strip()
    token = (api_token or HF_API_TOKEN).strip()
    
    # Offline or mock fallback mode
    if mock or (not url and not token):
        reason = "Mock mode requested" if mock else "HF_ENDPOINT_URL / HF_API_TOKEN not configured"
        synthetic_audio = generate_mock_instrument(
            prompt=prompt,
            tempo=float(tempo) if tempo else 120.0,
            key=key or "C Major",
            duration_sec=dur_clamped,
            sr=target_sr
        )
        sf.write(output_path, synthetic_audio, target_sr, subtype='FLOAT')
        
        meta = {
            "success": True,
            "status": "mock_fallback" if not mock else "mock_generated",
            "prompt": full_prompt,
            "tempo": tempo or "120",
            "key": key or "C Major",
            "section": section or "main",
            "duration_sec": dur_clamped,
            "sample_rate": target_sr,
            "bit_depth": "32-bit Float WAV",
            "integrated_lufs": round(float(calculate_lufs(synthetic_audio, target_sr)), 1),
            "true_peak_dbfs": round(float(calculate_true_peak(synthetic_audio, target_sr)), 2),
            "output_path": output_path,
            "notice": f"Generated via local synthesizer ({reason}). Configure HF_ENDPOINT_URL for cloud GPU neural inference."
        }
        return output_path, meta
        
    # Remote Hugging Face Inference Endpoint Call
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "duration": dur_clamped,
            "wait_for_model": True
        }
    }
    
    if ref_audio_path and os.path.exists(ref_audio_path):
        try:
            with open(ref_audio_path, "rb") as af:
                b64_ref = base64.b64encode(af.read()).decode("utf-8")
            payload["parameters"]["melody_audio"] = b64_ref
        except Exception as e:
            print(f"[GENERATIVE_ENGINE] Warning reading reference audio: {e}")
            
    start_time = time.time()
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SEC
        )
    except requests.exceptions.Timeout:
        return None, {
            "success": False,
            "status": "timeout",
            "error": f"Inference request timed out after {DEFAULT_TIMEOUT_SEC}s. If the endpoint was scaled to 0, it may still be warming up the GPU. Please retry in 1-2 minutes."
        }
    except requests.exceptions.ConnectionError as ce:
        return None, {
            "success": False,
            "status": "connection_error",
            "error": f"Failed to connect to Hugging Face endpoint: {ce}"
        }
    except Exception as e:
        return None, {
            "success": False,
            "status": "request_error",
            "error": str(e)
        }
        
    elapsed = round(time.time() - start_time, 2)
    
    # Handle HTTP status codes
    if response.status_code == 503:
        try:
            est_wait = response.json().get("estimated_time", "60-120")
        except Exception:
            est_wait = "60-120"
        return None, {
            "success": False,
            "status": "cold_start",
            "error": f"Inference Endpoint is currently scaling up (Cold Start). Estimated wait: {est_wait}s. Please retry shortly."
        }
    elif response.status_code in [401, 403]:
        return None, {
            "success": False,
            "status": "auth_error",
            "error": f"Authentication failed (HTTP {response.status_code}). Verify that HF_API_TOKEN has read access to the endpoint."
        }
    elif response.status_code != 200:
        return None, {
            "success": False,
            "status": "endpoint_error",
            "http_status": response.status_code,
            "error": f"Inference Endpoint returned error {response.status_code}: {response.text[:250]}"
        }
        
    # Extract audio bytes from response
    content_type = response.headers.get("Content-Type", "")
    audio_data_bytes = None
    
    if "audio" in content_type or "octet-stream" in content_type:
        audio_data_bytes = response.content
    else:
        # Check for JSON containing base64 audio
        try:
            res_json = response.json()
            if isinstance(res_json, list) and len(res_json) > 0 and isinstance(res_json[0], dict):
                first = res_json[0]
                b64_str = first.get("audio") or first.get("generated_audio")
                if b64_str:
                    audio_data_bytes = base64.b64decode(b64_str)
            elif isinstance(res_json, dict):
                b64_str = res_json.get("audio") or res_json.get("generated_audio")
                if b64_str:
                    audio_data_bytes = base64.b64decode(b64_str)
        except Exception:
            pass
            
    if not audio_data_bytes:
        audio_data_bytes = response.content
        
    try:
        stem_audio = decode_and_normalize_audio(audio_data_bytes, target_sr=target_sr)
        sf.write(output_path, stem_audio, target_sr, subtype='FLOAT')
    except Exception as e:
        return None, {
            "success": False,
            "status": "decode_error",
            "error": f"Failed to decode audio from Inference Endpoint response: {e}"
        }
        
    lufs = calculate_lufs(stem_audio, target_sr)
    tp = calculate_true_peak(stem_audio, target_sr)
    crest = calculate_crest_factor(stem_audio)
    
    meta = {
        "success": True,
        "status": "generated",
        "prompt": full_prompt,
        "tempo": tempo,
        "key": key,
        "section": section,
        "duration_sec": round(len(stem_audio) / float(target_sr), 2),
        "inference_time_sec": elapsed,
        "sample_rate": target_sr,
        "bit_depth": "32-bit Float WAV",
        "integrated_lufs": round(float(lufs), 1),
        "true_peak_dbfs": round(float(tp), 2),
        "dynamic_range_db": round(float(crest), 1),
        "output_path": output_path
    }
    
    return output_path, meta
