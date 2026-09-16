"""
Myooz Mixter Engine: Professional Vectorized AI & DSP Multi-Stem Mixing Studio
Author: Myooz Audio Intelligence Suite
Pure NumPy + Scipy signal implementation (ultra-fast, zero external C++ dependencies).
"""

import os
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal

def classify_stem(filename):
    """
    Intelligently categorizes a stem based on filename keywords
    Returns category dict with EQ profile, dynamics, pan position, and width.
    """
    name = filename.lower()
    
    # 1. Kick Drum
    if any(k in name for k in ["kick", "bombo", "bd"]):
        return {
            "type": "kick",
            "pan": 0.0,
            "width": 1.0,
            "gain_db": -0.5,
            "hpf": 32,
            "peak_freq": 58, "peak_gain": 2.5, "peak_q": 1.2,
            "mud_cut_freq": 250, "mud_cut_gain": -3.0,
            "click_shelf_freq": 3500, "click_shelf_gain": 2.0,
            "comp_thresh": -14.0, "comp_ratio": 3.8, "comp_attack_ms": 15, "comp_release_ms": 80
        }
        
    # 2. Bass / 808
    elif any(k in name for k in ["bass", "808", "sub", "bajo"]):
        return {
            "type": "bass",
            "pan": 0.0,
            "width": 0.8,
            "gain_db": -1.5,
            "hpf": 28,
            "peak_freq": 50, "peak_gain": 1.8, "peak_q": 1.0,
            "mud_cut_freq": 280, "mud_cut_gain": -2.5,
            "lpf": 4000,
            "sidechain_duck": True,
            "saturation_drive": 1.5
        }
        
    # 3. Lead Vocals
    elif any(k in name for k in ["lead", "voz lider", "vox lead", "principal", "acapella"]) or ("vocal" in name and not any(h in name for h in ["harm", "coro", "back"])):
        return {
            "type": "vocal_lead",
            "pan": 0.0,
            "width": 1.0,
            "gain_db": 0.5,
            "hpf": 85,
            "mud_cut_freq": 380, "mud_cut_gain": -2.5,
            "presence_freq": 3600, "presence_gain": 2.8, "presence_q": 1.1,
            "air_freq": 12000, "air_gain": 2.5,
            "comp_thresh": -16.0, "comp_ratio": 3.2, "comp_attack_ms": 20, "comp_release_ms": 120,
            "reverb_wet": 0.16
        }
        
    # 4. Backing Vocals / Harmonies
    elif any(k in name for k in ["vocal", "vox", "coro", "armonia", "harm", "back"]):
        return {
            "type": "vocal_back",
            "pan": 0.0,
            "width": 1.4,
            "gain_db": -2.5,
            "hpf": 130,
            "mud_cut_freq": 400, "mud_cut_gain": -2.5,
            "air_freq": 10000, "air_gain": 2.0,
            "comp_thresh": -15.0, "comp_ratio": 2.8, "comp_attack_ms": 25, "comp_release_ms": 140,
            "reverb_wet": 0.28
        }
        
    # 5. Claps / Snares
    elif any(k in name for k in ["clap", "snare", "caja", "tarola", "snap"]):
        pan_val = -0.5 if " l" in name or "_l" in name else (0.5 if " r" in name or "_r" in name else 0.0)
        return {
            "type": "snare_clap",
            "pan": pan_val,
            "width": 1.2,
            "gain_db": -1.8,
            "hpf": 180,
            "presence_freq": 1800, "presence_gain": 2.0, "presence_q": 1.2,
            "comp_thresh": -12.0, "comp_ratio": 3.0, "comp_attack_ms": 10, "comp_release_ms": 90
        }
        
    # 6. Maracas / Shakers / High Percussion
    elif any(k in name for k in ["maraca", "shaker", "hihat", "hat", "cymbal", "ride", "crash"]):
        return {
            "type": "percussion_high",
            "pan": 0.28,
            "width": 1.1,
            "gain_db": -2.2,
            "hpf": 320,
            "mud_cut_freq": 3800, "mud_cut_gain": -2.5,
            "air_freq": 11000, "air_gain": 2.2
        }
        
    # 7. Acoustic Strings (Cuatro, Guitar, Ukulele)
    elif any(k in name for k in ["cuatro", "guitar", "acustica", "tres", "cavaquinho"]):
        return {
            "type": "acoustic_strum",
            "pan": -0.22,
            "width": 1.25,
            "gain_db": -1.0,
            "hpf": 110,
            "peak_freq": 240, "peak_gain": 1.2, "peak_q": 1.0,
            "mud_cut_freq": 480, "mud_cut_gain": -2.0,
            "presence_freq": 2800, "presence_gain": 2.6, "presence_q": 1.1,
            "air_freq": 10000, "air_gain": 1.5,
            "comp_thresh": -16.0, "comp_ratio": 2.5, "comp_attack_ms": 30, "comp_release_ms": 140
        }
        
    # 8. Harp / Plucks / Keys
    elif any(k in name for k in ["arpa", "harp", "piano", "key", "teclado", "pluck"]):
        pan_val = -0.35 if " l" in name or "_l" in name else (0.35 if " r" in name or "_r" in name else -0.20)
        return {
            "type": "harp_keys",
            "pan": pan_val,
            "width": 1.15,
            "gain_db": -1.5,
            "hpf": 95,
            "peak_freq": 320, "peak_gain": 1.5, "peak_q": 1.0,
            "presence_freq": 5800, "presence_gain": 2.0, "presence_q": 1.2,
            "reverb_wet": 0.20
        }
        
    # 9. Violins / Orchestral Strings
    elif any(k in name for k in ["violin", "viola", "cello", "string", "cuerda", "orchestra"]):
        return {
            "type": "strings",
            "pan": 0.20,
            "width": 1.35,
            "gain_db": -1.8,
            "hpf": 140,
            "peak_freq": 460, "peak_gain": 1.5, "peak_q": 1.1,
            "mud_cut_freq": 3200, "mud_cut_gain": -2.0,
            "air_freq": 11000, "air_gain": 1.8,
            "reverb_wet": 0.22
        }
        
    # 10. Default / Instruments / FX
    else:
        return {
            "type": "other",
            "pan": 0.0,
            "width": 1.0,
            "gain_db": -2.0,
            "hpf": 80
        }

# --- High-Speed Vectorized DSP Primitives ---

def fast_biquad_peak(audio, freq, gain_db, sr=48000, q=1.0):
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
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    return signal.lfilter(b, a, audio, axis=0)

def fast_biquad_highpass(audio, cutoff, sr=48000, order=3):
    sos = signal.butter(order, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio, axis=0)

def fast_biquad_lowpass(audio, cutoff, sr=48000, order=2):
    sos = signal.butter(order, cutoff, btype='low', fs=sr, output='sos')
    return signal.sosfilt(sos, audio, axis=0)

def fast_biquad_highshelf(audio, freq, gain_db, sr=48000):
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
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    return signal.lfilter(b, a, audio, axis=0)

def fast_compressor(audio, sr=48000, threshold_db=-16.0, ratio=3.0, attack_ms=20.0, release_ms=100.0, makeup_db=0.0):
    """Vectorized C-speed feedforward compressor using 1-pole recursive filtering"""
    sidechain = np.max(np.abs(audio), axis=1) if audio.ndim == 2 else np.abs(audio)
    rel_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
    b = [1.0 - rel_coef]
    a = [1.0, -rel_coef]
    env = signal.lfilter(b, a, sidechain)
    
    env_db = 20 * np.log10(env + 1e-9)
    gain_db = np.zeros_like(env_db)
    over = env_db > threshold_db
    gain_db[over] = (threshold_db + (env_db[over] - threshold_db) / ratio) - env_db[over]
    
    gain_lin = 10 ** ((gain_db + makeup_db) / 20.0)
    if audio.ndim == 2:
        return audio * gain_lin[:, np.newaxis]
    return audio * gain_lin

def fast_sidechain_duck(audio, key_audio, sr=48000, threshold_db=-16.0, duck_db=-2.8, release_ms=110.0):
    """Sidechain ducking: ducks bass/808 when kick hits"""
    key_sc = np.max(np.abs(key_audio), axis=1) if key_audio.ndim == 2 else np.abs(key_audio)
    rel_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
    b = [1.0 - rel_coef]
    a = [1.0, -rel_coef]
    key_env = signal.lfilter(b, a, key_sc)
    
    thresh_lin = 10 ** (threshold_db / 20.0)
    duck_lin = 10 ** (duck_db / 20.0)
    
    # Gain envelope: 1.0 when quiet, duck_lin when triggered
    gain_target = np.where(key_env > thresh_lin, duck_lin, 1.0)
    gain_curve = signal.lfilter(b, a, gain_target)
    
    if audio.ndim == 2:
        return audio * gain_curve[:, np.newaxis]
    return audio * gain_curve

def fast_stereo_pan(audio_stereo, pan_pos=0.0):
    if abs(pan_pos) < 0.02:
        return audio_stereo
    angle = (pan_pos + 1.0) * (math.pi / 4.0)
    left_gain = math.cos(angle) * math.sqrt(2)
    right_gain = math.sin(angle) * math.sqrt(2)
    out = np.zeros_like(audio_stereo)
    out[:, 0] = audio_stereo[:, 0] * left_gain
    out[:, 1] = audio_stereo[:, 1] * right_gain
    return out

def fast_stereo_width(audio_stereo, width=1.0):
    if abs(width - 1.0) < 0.05:
        return audio_stereo
    mid = 0.5 * (audio_stereo[:, 0] + audio_stereo[:, 1])
    side = 0.5 * (audio_stereo[:, 0] - audio_stereo[:, 1]) * width
    out = np.zeros_like(audio_stereo)
    out[:, 0] = mid + side
    out[:, 1] = mid - side
    return out

def fast_reverb_send(audio, sr=48000, wet=0.18):
    """Lightweight comb-allpass spatial reverberation"""
    if wet <= 0.01:
        return audio
    delays = [int(sr * d) for d in [0.031, 0.037, 0.043]]
    mono_in = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    
    wet_accum = np.zeros(len(mono_in))
    for d in delays:
        b = [0.0] * d + [0.35]
        a = [1.0] + [0.0] * (d - 1) + [-0.35]
        comb = signal.lfilter(b, a, mono_in)
        wet_accum += comb
        
    wet_accum = wet_accum / len(delays)
    # Decorrelate for stereo
    delay_decorr = int(sr * 0.012)
    wet_l = wet_accum
    wet_r = np.roll(wet_accum, delay_decorr)
    wet_stereo = np.column_stack([wet_l, wet_r])
    
    dry_stereo = audio if (audio.ndim == 2 and audio.shape[1] == 2) else np.column_stack([audio, audio])
    return dry_stereo + wet * wet_stereo

# ==========================================
# MAIN MIX PROCESSOR
# ==========================================

def process_and_mix_stems(stem_paths, output_path=None, mix_style="modern", vocal_fx_level=0.3):
    """
    Mixes multiple stems with intelligent channel strips, sidechain ducking,
    stereo placement, and headroom calibration.
    """
    if not stem_paths:
        raise ValueError("No stems provided to mix.")
        
    if not output_path:
        first_dir = os.path.dirname(stem_paths[0]) or "."
        output_path = os.path.join(first_dir, "mixter_final_mix.wav")
        
    sr = 48000
    
    # 1. Load files and determine max timeline
    max_len = 0
    loaded_stems = []
    kick_data = None
    
    for item in stem_paths:
        if isinstance(item, (tuple, list)):
            p = item[0]
            fname = item[1] if len(item) > 1 else os.path.basename(p)
        else:
            p = str(item)
            fname = os.path.basename(p)
            
        data, file_sr = sf.read(p)
        if file_sr != sr:
            # Resample to 48kHz
            gcd = math.gcd(sr, file_sr)
            up = sr // gcd
            down = file_sr // gcd
            data = signal.resample_poly(data, up, down, axis=0)
            
        if data.ndim == 1:
            data_stereo = np.column_stack([data, data])
        elif data.shape[1] == 1:
            data_stereo = np.repeat(data, 2, axis=1)
        else:
            data_stereo = data[:, :2]
            
        if len(data_stereo) > max_len:
            max_len = len(data_stereo)
            
        conf = classify_stem(fname)
        
        stem_obj = {
            "name": fname,
            "data": data_stereo,
            "conf": conf
        }
        loaded_stems.append(stem_obj)
        
        if conf["type"] == "kick" and kick_data is None:
            kick_data = data_stereo
            
    # Pad all to max_len
    for s in loaded_stems:
        if len(s["data"]) < max_len:
            pad = np.zeros((max_len - len(s["data"]), 2), dtype=s["data"].dtype)
            s["data"] = np.vstack([s["data"], pad])
            
    if kick_data is not None and len(kick_data) < max_len:
        pad_k = np.zeros((max_len - len(kick_data)), dtype=kick_data.dtype)
        kick_mono = np.mean(kick_data, axis=1)
        kick_data = np.concatenate([kick_mono, pad_k])
    elif kick_data is not None:
        kick_data = np.mean(kick_data[:max_len], axis=1)
        
    # 2. Process each channel strip
    summing_bus = np.zeros((max_len, 2), dtype=np.float64)
    
    for s in loaded_stems:
        data = s["data"].astype(np.float64)
        conf = s["conf"]
        t = conf["type"]
        
        # Highpass Filter
        if "hpf" in conf:
            data = fast_biquad_highpass(data, conf["hpf"], sr)
            
        # Parametric EQ Peaking
        if "peak_freq" in conf:
            data = fast_biquad_peak(data, conf["peak_freq"], conf["peak_gain"], sr, q=conf.get("peak_q", 1.0))
            
        # Mud Cut
        if "mud_cut_freq" in conf:
            data = fast_biquad_peak(data, conf["mud_cut_freq"], conf["mud_cut_gain"], sr, q=1.2)
            
        # Presence
        if "presence_freq" in conf:
            data = fast_biquad_peak(data, conf["presence_freq"], conf["presence_gain"], sr, q=conf.get("presence_q", 1.0))
            
        # Air Shelf
        if "air_freq" in conf:
            data = fast_biquad_highshelf(data, conf["air_freq"], conf["air_gain"], sr)
            
        # Lowpass filter
        if "lpf" in conf:
            data = fast_biquad_lowpass(data, conf["lpf"], sr)
            
        # Dynamic Compression
        if "comp_thresh" in conf:
            data = fast_compressor(
                data, sr,
                threshold_db=conf["comp_thresh"],
                ratio=conf.get("comp_ratio", 3.0),
                attack_ms=conf.get("comp_attack_ms", 20),
                release_ms=conf.get("comp_release_ms", 100),
                makeup_db=1.0
            )
            
        # Sidechain Ducking under Kick (for bass/808)
        if conf.get("sidechain_duck") and kick_data is not None:
            data = fast_sidechain_duck(data, kick_data, sr, threshold_db=-18.0, duck_db=-2.8, release_ms=110.0)
            
        # Vocal Reverb Send
        wet_amount = conf.get("reverb_wet", 0.0)
        if vocal_fx_level is not None and "vocal" in t:
            wet_amount = wet_amount * float(vocal_fx_level) * 2.0
        if wet_amount > 0.02:
            data = fast_reverb_send(data, sr, wet=wet_amount)
            
        # Stereo Panning & Width
        if conf.get("width", 1.0) != 1.0:
            data = fast_stereo_width(data, conf["width"])
        if conf.get("pan", 0.0) != 0.0:
            data = fast_stereo_pan(data, conf["pan"])
            
        # Channel Fader Gain
        gain_lin = 10 ** (conf.get("gain_db", 0.0) / 20.0)
        data = data * gain_lin
        
        # Sum to master bus
        summing_bus += data
        
    # 3. Headroom Calibration to -6.0 dBFS Peak
    peak_val = np.max(np.abs(summing_bus))
    if peak_val > 1e-6:
        target_peak_lin = 10 ** (-6.0 / 20.0) # 0.501
        summing_bus = summing_bus * (target_peak_lin / peak_val)
        
    # Export 32-bit Float WAV (Studio Standard)
    sf.write(output_path, summing_bus.astype(np.float32), sr, subtype='FLOAT')
    
    report = {
        "stems_count": len(stem_paths),
        "mix_style": mix_style,
        "headroom": "-6.0 dBFS Peak (32-bit Float Calibrated)",
        "output_path": output_path
    }
    return output_path, report
