"""
Myooz Mixter Engine 2026: Studio Multi-Stem AI Mixing & Dynamic Resonance Suppression
Author: Myooz Audio Intelligence Suite
Pure NumPy + Scipy signal implementation (ultra-fast, zero external C++ dependencies).
"""

import os
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal

# ==========================================
# 2026 GENRE & DERIVATIVE SUBGENRE MATRIX
# ==========================================

MIX_GENRES = {
    "urbano": {
        "name": "Reggaeton / Urbano & Latin Trap",
        "default_subgenre": "neo_perreo",
        "subgenres": {
            "neo_perreo": {
                "name": "Neo-Perreo / Dembow 2026",
                "desc": "Subgrave pegador, voz al frente seca y nítida, claps estéreo brillantes",
                "kick_gain": 0.8, "bass_gain": 0.2, "vocal_gain": 2.2, "snare_gain": -0.8, "other_gain": -2.5,
                "sub_boost_freq": 54, "sub_boost_db": 3.0,
                "vocal_air_db": 3.2, "reverb_wet": 0.12, "sidechain_duck_db": -3.8,
                "width_drums": 1.1, "width_instruments": 1.25, "width_vocals": 1.0
            },
            "trap_latino": {
                "name": "Trap Latino 808",
                "desc": "Sub 808 colosal, medios limpios, hi-hats quirúrgicos con apertura",
                "kick_gain": 0.2, "bass_gain": 1.8, "vocal_gain": 2.4, "snare_gain": -0.5, "other_gain": -3.0,
                "sub_boost_freq": 44, "sub_boost_db": 3.8,
                "vocal_air_db": 3.5, "reverb_wet": 0.15, "sidechain_duck_db": -4.2,
                "width_drums": 1.15, "width_instruments": 1.35, "width_vocals": 1.0
            },
            "afrobeat_urbano": {
                "name": "Afrobeat / Dancehall Urbano",
                "desc": "Bajo cálido y fluido, percusiones orgánicas abiertas y reverb envolvente",
                "kick_gain": -0.5, "bass_gain": 0.8, "vocal_gain": 1.8, "snare_gain": -1.2, "other_gain": -1.5,
                "sub_boost_freq": 58, "sub_boost_db": 2.4,
                "vocal_air_db": 2.6, "reverb_wet": 0.24, "sidechain_duck_db": -2.8,
                "width_drums": 1.3, "width_instruments": 1.4, "width_vocals": 1.05
            }
        }
    },
    "electronic": {
        "name": "Electronic / EDM / Club",
        "default_subgenre": "tech_house",
        "subgenres": {
            "tech_house": {
                "name": "Tech House / Peak Time",
                "desc": "Bombo sólido a 55Hz, bajo rodante en 150Hz, claps anchos y rítmica marcada",
                "kick_gain": 1.2, "bass_gain": 0.6, "vocal_gain": 0.8, "snare_gain": 0.0, "other_gain": -1.5,
                "sub_boost_freq": 55, "sub_boost_db": 3.2,
                "vocal_air_db": 2.2, "reverb_wet": 0.18, "sidechain_duck_db": -4.5,
                "width_drums": 1.2, "width_instruments": 1.35, "width_vocals": 1.1
            },
            "modern_edm": {
                "name": "Modern EDM / Festival",
                "desc": "Máxima energía y densidad, leads potentes, sidechain agresivo",
                "kick_gain": 1.4, "bass_gain": 1.0, "vocal_gain": 1.6, "snare_gain": 0.6, "other_gain": 0.0,
                "sub_boost_freq": 50, "sub_boost_db": 3.6,
                "vocal_air_db": 3.0, "reverb_wet": 0.26, "sidechain_duck_db": -5.0,
                "width_drums": 1.25, "width_instruments": 1.45, "width_vocals": 1.15
            },
            "melodic_techno": {
                "name": "Melodic Techno / Deep",
                "desc": "Sub bajos hipnóticos, percusiones espaciales, texturas etéreas",
                "kick_gain": 0.6, "bass_gain": 1.0, "vocal_gain": 0.2, "snare_gain": -0.8, "other_gain": 0.5,
                "sub_boost_freq": 48, "sub_boost_db": 2.8,
                "vocal_air_db": 2.0, "reverb_wet": 0.28, "sidechain_duck_db": -3.8,
                "width_drums": 1.2, "width_instruments": 1.4, "width_vocals": 1.2
            }
        }
    },
    "pop": {
        "name": "Pop / R&B Commercial",
        "default_subgenre": "billboard_pop",
        "subgenres": {
            "billboard_pop": {
                "name": "Billboard Pop 2026",
                "desc": "Voz estelar cristalina al frente (+2.8dB), compresión paralela y brillo aéreo",
                "kick_gain": 0.0, "bass_gain": -0.4, "vocal_gain": 2.8, "snare_gain": -0.5, "other_gain": -2.0,
                "sub_boost_freq": 60, "sub_boost_db": 2.0,
                "vocal_air_db": 3.6, "reverb_wet": 0.20, "sidechain_duck_db": -2.5,
                "width_drums": 1.15, "width_instruments": 1.35, "width_vocals": 1.0
            },
            "dark_rnb": {
                "name": "Modern R&B / Soul",
                "desc": "Graves profundos aterciopelados, calidez en teclados y voz íntima",
                "kick_gain": -0.4, "bass_gain": 1.2, "vocal_gain": 2.2, "snare_gain": -1.2, "other_gain": -1.8,
                "sub_boost_freq": 42, "sub_boost_db": 2.6,
                "vocal_air_db": 2.5, "reverb_wet": 0.24, "sidechain_duck_db": -3.0,
                "width_drums": 1.1, "width_instruments": 1.3, "width_vocals": 1.0
            },
            "synthpop": {
                "name": "Synthpop / Retrowave",
                "desc": "Cajas con reverb de placa estéreo, sintetizadores anchos y bajos ochenteros",
                "kick_gain": 0.6, "bass_gain": 0.2, "vocal_gain": 2.0, "snare_gain": 0.5, "other_gain": -0.4,
                "sub_boost_freq": 65, "sub_boost_db": 2.2,
                "vocal_air_db": 3.0, "reverb_wet": 0.28, "sidechain_duck_db": -2.6,
                "width_drums": 1.2, "width_instruments": 1.4, "width_vocals": 1.1
            }
        }
    },
    "hiphop": {
        "name": "Hip-Hop / Drill / Trap",
        "default_subgenre": "atlanta_trap",
        "subgenres": {
            "atlanta_trap": {
                "name": "Atlanta 808 Trap",
                "desc": "808s masivos mono-sub, platos afilados y voz cortante y definida",
                "kick_gain": 0.5, "bass_gain": 2.0, "vocal_gain": 2.2, "snare_gain": 0.0, "other_gain": -2.5,
                "sub_boost_freq": 42, "sub_boost_db": 3.8,
                "vocal_air_db": 3.2, "reverb_wet": 0.14, "sidechain_duck_db": -4.2,
                "width_drums": 1.15, "width_instruments": 1.3, "width_vocals": 1.0
            },
            "uk_drill": {
                "name": "UK Drill / Sliding Sub",
                "desc": "Bajos deslizantes, hi-hats sincopados en estéreo y distorsión armónica sutil",
                "kick_gain": 0.2, "bass_gain": 1.6, "vocal_gain": 1.8, "snare_gain": 0.2, "other_gain": -3.0,
                "sub_boost_freq": 46, "sub_boost_db": 3.5,
                "vocal_air_db": 3.2, "reverb_wet": 0.12, "sidechain_duck_db": -4.0,
                "width_drums": 1.2, "width_instruments": 1.25, "width_vocals": 1.0
            },
            "boombap": {
                "name": "East Coast Boom Bap",
                "desc": "Textura analógica de cinta, caja crujiente en 200Hz y balance orgánico",
                "kick_gain": 0.6, "bass_gain": 0.6, "vocal_gain": 2.0, "snare_gain": 0.6, "other_gain": -1.5,
                "sub_boost_freq": 68, "sub_boost_db": 2.0,
                "vocal_air_db": 1.5, "reverb_wet": 0.16, "sidechain_duck_db": -2.4,
                "width_drums": 1.05, "width_instruments": 1.2, "width_vocals": 1.0
            }
        }
    },
    "rock": {
        "name": "Rock / Indie / Alternative",
        "default_subgenre": "modern_rock",
        "subgenres": {
            "modern_rock": {
                "name": "Modern High-Gain Rock",
                "desc": "Pared de guitarras abiertas L/R, batería con pegada y bajo centrado",
                "kick_gain": 0.8, "bass_gain": 0.2, "vocal_gain": 1.2, "snare_gain": 0.8, "other_gain": 1.0,
                "sub_boost_freq": 65, "sub_boost_db": 2.0,
                "vocal_air_db": 2.2, "reverb_wet": 0.16, "sidechain_duck_db": -2.0,
                "width_drums": 1.2, "width_instruments": 1.45, "width_vocals": 1.05
            },
            "indie_pop": {
                "name": "Indie / Bedroom Pop",
                "desc": "Guitarras chorus espaciales, batería orgánica con dinámica natural",
                "kick_gain": -0.2, "bass_gain": 0.0, "vocal_gain": 1.5, "snare_gain": -0.5, "other_gain": 0.0,
                "sub_boost_freq": 60, "sub_boost_db": 1.5,
                "vocal_air_db": 2.0, "reverb_wet": 0.24, "sidechain_duck_db": -2.2,
                "width_drums": 1.15, "width_instruments": 1.35, "width_vocals": 1.1
            },
            "pop_punk": {
                "name": "Pop Punk / Emo Rock",
                "desc": "Ataque rápido en transitorios, caja contundente y voces enérgicas",
                "kick_gain": 1.0, "bass_gain": 0.4, "vocal_gain": 1.6, "snare_gain": 1.0, "other_gain": 0.8,
                "sub_boost_freq": 70, "sub_boost_db": 2.0,
                "vocal_air_db": 2.8, "reverb_wet": 0.15, "sidechain_duck_db": -2.2,
                "width_drums": 1.2, "width_instruments": 1.4, "width_vocals": 1.0
            }
        }
    },
    "acoustic": {
        "name": "Acústico / Regional / Folk",
        "default_subgenre": "regional_tumbado",
        "subgenres": {
            "regional_tumbado": {
                "name": "Regional Urbano / Corridos Tumbados",
                "desc": "Requintos y guitarras de 12 cuerdas cristalinas, tololoche contundente y voz al frente",
                "kick_gain": -0.5, "bass_gain": 1.2, "vocal_gain": 2.4, "snare_gain": -0.5, "other_gain": 0.8,
                "sub_boost_freq": 52, "sub_boost_db": 2.6,
                "vocal_air_db": 3.0, "reverb_wet": 0.18, "sidechain_duck_db": -2.0,
                "width_drums": 1.15, "width_instruments": 1.35, "width_vocals": 1.0
            },
            "acoustic_songwriter": {
                "name": "Acoustic Singer-Songwriter",
                "desc": "Cero bombeo, máxima fidelidad en madera y cuerdas con acústica de sala real",
                "kick_gain": -1.0, "bass_gain": -0.5, "vocal_gain": 2.0, "snare_gain": -1.0, "other_gain": 0.5,
                "sub_boost_freq": 80, "sub_boost_db": 1.0,
                "vocal_air_db": 2.0, "reverb_wet": 0.22, "sidechain_duck_db": -1.5,
                "width_drums": 1.05, "width_instruments": 1.25, "width_vocals": 1.0
            },
            "lofi_chill": {
                "name": "Lo-Fi Chill Beats",
                "desc": "Filtro cálido de frecuencias altas, compresión suave y relajada",
                "kick_gain": 0.0, "bass_gain": 0.6, "vocal_gain": 1.0, "snare_gain": -1.0, "other_gain": -0.5,
                "sub_boost_freq": 55, "sub_boost_db": 2.0,
                "vocal_air_db": -1.2, "reverb_wet": 0.20, "sidechain_duck_db": -3.0,
                "width_drums": 1.05, "width_instruments": 1.2, "width_vocals": 1.0
            }
        }
    }
}

OLD_STYLE_MAP = {
    "modern": ("urbano", "neo_perreo"),
    "punchy": ("electronic", "tech_house"),
    "club": ("electronic", "tech_house"),
    "acoustic": ("acoustic", "acoustic_songwriter"),
    "streaming": ("pop", "billboard_pop"),
    "dynamic": ("rock", "indie_pop")
}

def classify_stem(filename):
    name = filename.lower()
    if any(k in name for k in ["kick", "bombo", "bd"]):
        return {
            "type": "kick", "pan": 0.0, "width": 1.0, "gain_db": -0.5,
            "hpf": 32, "peak_freq": 58, "peak_gain": 2.5, "peak_q": 1.2,
            "mud_cut_freq": 250, "mud_cut_gain": -3.0,
            "click_shelf_freq": 3500, "click_shelf_gain": 2.0,
            "comp_thresh": -14.0, "comp_ratio": 3.8, "comp_attack_ms": 15, "comp_release_ms": 80
        }
    elif any(k in name for k in ["bass", "808", "sub", "bajo"]):
        return {
            "type": "bass", "pan": 0.0, "width": 0.8, "gain_db": -1.5,
            "hpf": 28, "peak_freq": 50, "peak_gain": 1.8, "peak_q": 1.0,
            "mud_cut_freq": 280, "mud_cut_gain": -2.5, "lpf": 4500,
            "sidechain_duck": True, "saturation_drive": 1.5
        }
    elif any(k in name for k in ["lead", "voz lider", "vox lead", "principal", "acapella"]) or ("vocal" in name and not any(h in name for h in ["harm", "coro", "back"])):
        return {
            "type": "vocal_lead", "pan": 0.0, "width": 1.0, "gain_db": 0.5,
            "hpf": 85, "mud_cut_freq": 380, "mud_cut_gain": -2.5,
            "presence_freq": 3600, "presence_gain": 2.8, "presence_q": 1.1,
            "air_freq": 12000, "air_gain": 2.5,
            "comp_thresh": -16.0, "comp_ratio": 3.2, "comp_attack_ms": 20, "comp_release_ms": 120,
            "reverb_wet": 0.16
        }
    elif any(k in name for k in ["vocal", "vox", "coro", "armonia", "harm", "back"]):
        return {
            "type": "vocal_back", "pan": 0.0, "width": 1.4, "gain_db": -2.5,
            "hpf": 130, "mud_cut_freq": 400, "mud_cut_gain": -2.5,
            "air_freq": 10000, "air_gain": 2.0,
            "comp_thresh": -15.0, "comp_ratio": 2.8, "comp_attack_ms": 25, "comp_release_ms": 140,
            "reverb_wet": 0.28
        }
    elif any(k in name for k in ["clap", "snare", "caja", "tarola", "snap"]):
        pan_val = -0.4 if " l" in name or "_l" in name else (0.4 if " r" in name or "_r" in name else 0.0)
        return {
            "type": "snare_clap", "pan": pan_val, "width": 1.2, "gain_db": -1.8,
            "hpf": 160, "presence_freq": 2000, "presence_gain": 2.2, "presence_q": 1.2,
            "comp_thresh": -12.0, "comp_ratio": 3.0, "comp_attack_ms": 10, "comp_release_ms": 90
        }
    elif any(k in name for k in ["maraca", "shaker", "hihat", "hat", "cymbal", "ride", "crash", "perc"]):
        return {
            "type": "percussion_high", "pan": 0.25, "width": 1.15, "gain_db": -2.2,
            "hpf": 300, "mud_cut_freq": 3800, "mud_cut_gain": -2.5, "air_freq": 11000, "air_gain": 2.2
        }
    elif any(k in name for k in ["cuatro", "guitar", "acustica", "tres", "cavaquinho"]):
        return {
            "type": "acoustic_strum", "pan": -0.20, "width": 1.25, "gain_db": -1.0,
            "hpf": 105, "peak_freq": 240, "peak_gain": 1.2, "peak_q": 1.0,
            "mud_cut_freq": 480, "mud_cut_gain": -2.0,
            "presence_freq": 2800, "presence_gain": 2.6, "presence_q": 1.1,
            "air_freq": 10000, "air_gain": 1.5,
            "comp_thresh": -16.0, "comp_ratio": 2.5, "comp_attack_ms": 30, "comp_release_ms": 140
        }
    elif any(k in name for k in ["arpa", "harp", "piano", "key", "teclado", "pluck", "synth"]):
        pan_val = -0.30 if " l" in name or "_l" in name else (0.30 if " r" in name or "_r" in name else -0.15)
        return {
            "type": "harp_keys", "pan": pan_val, "width": 1.2, "gain_db": -1.5,
            "hpf": 95, "peak_freq": 320, "peak_gain": 1.5, "peak_q": 1.0,
            "presence_freq": 5800, "presence_gain": 2.0, "presence_q": 1.2,
            "reverb_wet": 0.20
        }
    elif any(k in name for k in ["violin", "viola", "cello", "string", "cuerda", "orchestra"]):
        return {
            "type": "strings", "pan": 0.20, "width": 1.35, "gain_db": -1.8,
            "hpf": 130, "peak_freq": 460, "peak_gain": 1.5, "peak_q": 1.1,
            "mud_cut_freq": 3200, "mud_cut_gain": -2.0,
            "air_freq": 11000, "air_gain": 1.8, "reverb_wet": 0.22
        }
    else:
        return {
            "type": "other", "pan": 0.0, "width": 1.0, "gain_db": -2.0, "hpf": 80
        }

# Vectorized DSP Primitives
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

def fast_sidechain_duck(audio, key_audio, sr=48000, threshold_db=-16.0, duck_db=-3.5, release_ms=110.0):
    key_sc = np.max(np.abs(key_audio), axis=1) if key_audio.ndim == 2 else np.abs(key_audio)
    rel_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
    b = [1.0 - rel_coef]
    a = [1.0, -rel_coef]
    key_env = signal.lfilter(b, a, key_sc)
    
    thresh_lin = 10 ** (threshold_db / 20.0)
    duck_lin = 10 ** (duck_db / 20.0)
    
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

# Impulse Response for FFT Plate Reverb
_IR_SR = 48000
_IR_LEN = int(_IR_SR * 0.75)
_T_IR = np.linspace(0, 0.75, _IR_LEN, dtype=np.float32)
_DECAY = np.exp(-_T_IR * 6.0).astype(np.float32)
np.random.seed(42)
_IR_L = (np.random.randn(_IR_LEN).astype(np.float32) * _DECAY) * 0.05
_IR_R = (np.random.randn(_IR_LEN).astype(np.float32) * _DECAY) * 0.05

def fast_reverb_send(audio, sr=48000, wet=0.18):
    if wet <= 0.01:
        return audio
    mono_in = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    wet_l = signal.fftconvolve(mono_in, _IR_L, mode='same')
    wet_r = signal.fftconvolve(mono_in, _IR_R, mode='same')
    wet_stereo = np.column_stack([wet_l, wet_r])
    dry_stereo = audio if (audio.ndim == 2 and audio.shape[1] == 2) else np.column_stack([audio, audio])
    return dry_stereo + wet * wet_stereo

# Dynamic Resonance Suppression & Unmasking
def suppress_harsh_resonances(audio, sr=48000, sensitivity=1.0):
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    if len(mono) < 4096 or np.max(np.abs(mono)) < 1e-4:
        return audio, []
        
    nperseg = min(4096, len(mono))
    f, psd = signal.welch(mono, fs=sr, nperseg=nperseg)
    
    kernel = 41
    if len(psd) < kernel:
        return audio, []
    smooth_psd = signal.medfilt(psd, kernel_size=kernel)
    
    diff_db = 10.0 * np.log10(np.maximum(psd, 1e-12) / np.maximum(smooth_psd, 1e-12))
    
    tamed_resonances = []
    cleaned = audio.copy()
    
    bands = [
        (250, 500, 4.0),
        (2200, 4500, 4.2)
    ]
    
    for f_low, f_high, thresh in bands:
        mask = (f >= f_low) & (f <= f_high)
        if not np.any(mask):
            continue
            
        band_diff = diff_db[mask]
        band_f = f[mask]
        
        peak_idx = np.argmax(band_diff)
        max_diff = band_diff[peak_idx]
        
        threshold = thresh / max(0.5, sensitivity)
        if max_diff > threshold:
            peak_freq = band_f[peak_idx]
            attenuation_db = -float(min(5.5, max_diff * 0.75))
            cleaned = fast_biquad_peak(cleaned, freq=float(peak_freq), gain_db=attenuation_db, sr=sr, q=4.5)
            tamed_resonances.append({
                "freq_hz": round(float(peak_freq)),
                "attenuation_db": round(attenuation_db, 1)
            })
            
    return cleaned, tamed_resonances

def apply_spectral_unmasking(instrument_audio, vocal_audio, sr=48000):
    if vocal_audio is None or instrument_audio is None:
        return instrument_audio
        
    vox_mono = np.mean(vocal_audio, axis=1) if vocal_audio.ndim == 2 else vocal_audio
    vocal_presence = fast_biquad_peak(vox_mono, freq=2400, gain_db=5.0, sr=sr, q=1.0)
    
    rel_coef = np.exp(-1.0 / (sr * (120.0 / 1000.0)))
    b = [1.0 - rel_coef]
    a = [1.0, -rel_coef]
    vocal_env = signal.lfilter(b, a, np.abs(vocal_presence))
    
    vocal_peak = np.max(vocal_env)
    if vocal_peak < 1e-4:
        return instrument_audio
        
    thresh = vocal_peak * 0.15
    active_vocal = vocal_env > thresh
    
    if not np.any(active_vocal):
        return instrument_audio
        
    carved = fast_biquad_peak(instrument_audio, freq=2200, gain_db=-2.2, sr=sr, q=1.2)
    gain_curve = signal.lfilter(b, a, active_vocal.astype(np.float64))[:, np.newaxis]
    return instrument_audio * (1.0 - gain_curve) + carved * gain_curve

# Pure SciPy ITU-R BS.1770-4 K-Weighting Metrics (Zero External Dependencies)
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
        return round(float(gamma_r), 1)
        
    integrated_lufs = -0.691 + 10 * np.log10(np.mean(block_powers[idx_rel]) + 1e-12)
    return round(float(integrated_lufs), 1)

def calculate_true_peak(audio, sr=44100):
    sample_peak = np.max(np.abs(audio))
    if sample_peak < 1e-6:
        return -70.0
    # Evaluate True-Peak using localized 4x oversampling on the peak window (<0.02s instead of 25s)
    mono = np.max(np.abs(audio), axis=1) if audio.ndim == 2 else np.abs(audio)
    top_idx = int(np.argmax(mono))
    window = int(sr * 2)
    start_idx = max(0, top_idx - window)
    end_idx = min(len(audio), top_idx + window)
    slice_audio = audio[start_idx:end_idx]
    if len(slice_audio) > 0:
        audio_4x = signal.resample_poly(slice_audio, 4, 1, axis=0)
        peak = max(sample_peak, float(np.max(np.abs(audio_4x))))
    else:
        peak = sample_peak
    return round(float(20 * np.log10(peak + 1e-12)), 2)

def calculate_crest_factor(audio):
    peak = np.max(np.abs(audio)) + 1e-9
    rms = np.sqrt(np.mean(audio**2)) + 1e-9
    return round(float(20 * np.log10(peak / rms)), 1)

# Main Mix Function
def process_and_mix_stems(stem_paths, output_path=None, mix_style="urbano", subgenre=None, vocal_fx_level=0.3, resonance_suppression=True, **kwargs):
    if not stem_paths:
        raise ValueError("No stems provided to mix.")
        
    if not output_path:
        first_item = stem_paths[0]
        first_p = first_item[0] if isinstance(first_item, (tuple, list)) else str(first_item)
        first_dir = os.path.dirname(first_p) or "/tmp"
        output_path = os.path.join(first_dir, "mixter_final_mix.wav")
        
    # Detect native sample rate from stems (Demucs outputs 44100 Hz, avoids 140s of resample_poly)
    sr = 44100
    for item in stem_paths:
        p = item[0] if isinstance(item, (tuple, list)) else str(item)
        try:
            info = sf.info(p)
            sr = info.samplerate
            break
        except Exception:
            pass
    
    # 1. Resolve genre and subgenre
    genre_key = str(mix_style).lower().strip()
    subgenre_key = str(subgenre).lower().strip() if subgenre else None
    
    if genre_key in OLD_STYLE_MAP:
        genre_key, default_sub = OLD_STYLE_MAP[genre_key]
        if not subgenre_key:
            subgenre_key = default_sub
            
    if genre_key not in MIX_GENRES:
        genre_key = "urbano"
        
    genre_data = MIX_GENRES[genre_key]
    if not subgenre_key or subgenre_key not in genre_data["subgenres"]:
        subgenre_key = genre_data.get("default_subgenre", list(genre_data["subgenres"].keys())[0])
        
    preset = genre_data["subgenres"][subgenre_key]
    
    # 2. Load files and determine max timeline
    max_len = 0
    loaded_stems = []
    kick_data = None
    vocal_data = None
    
    for item in stem_paths:
        if isinstance(item, (tuple, list)):
            p = item[0]
            fname = item[1] if len(item) > 1 else os.path.basename(p)
        else:
            p = str(item)
            fname = os.path.basename(p)
            
        data, file_sr = sf.read(p, always_2d=True, dtype='float32')
        if file_sr != sr:
            gcd = math.gcd(sr, file_sr)
            data = signal.resample_poly(data, sr // gcd, file_sr // gcd, axis=0)
            
        if data.shape[1] == 1:
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
        elif conf["type"] == "vocal_lead" and vocal_data is None:
            vocal_data = data_stereo
            
    # Pad all stems to max_len
    for s in loaded_stems:
        if len(s["data"]) < max_len:
            pad = np.zeros((max_len - len(s["data"]), 2), dtype=s["data"].dtype)
            s["data"] = np.vstack([s["data"], pad])
            
    if kick_data is not None and len(kick_data) < max_len:
        pad_k = np.zeros((max_len - len(kick_data)), dtype=kick_data.dtype)
        kick_mono = np.mean(kick_data, axis=1)
        kick_data = np.column_stack([np.concatenate([kick_mono, pad_k]), np.concatenate([kick_mono, pad_k])])
    elif kick_data is not None:
        kick_data = kick_data[:max_len]
        
    if vocal_data is not None and len(vocal_data) < max_len:
        pad_v = np.zeros((max_len - len(vocal_data), 2), dtype=vocal_data.dtype)
        vocal_data = np.vstack([vocal_data, pad_v])
    elif vocal_data is not None:
        vocal_data = vocal_data[:max_len]
        
    # 3. Process each channel strip with 2026 Genre Profile & Anti-Resonance
    summing_bus = np.zeros((max_len, 2), dtype=np.float64)
    raw_sum = np.zeros((max_len, 2), dtype=np.float64)
    all_tamed_resonances = []
    
    for s in loaded_stems:
        data = s["data"].astype(np.float64)
        raw_sum += data
        conf = s["conf"]
        t = conf["type"]
        
        # Step A: Dynamic Resonance Suppression
        if resonance_suppression and t in ["vocal_lead", "vocal_back", "acoustic_strum", "strings", "harp_keys", "other"]:
            data, tamed = suppress_harsh_resonances(data, sr=sr, sensitivity=1.1)
            for item in tamed:
                all_tamed_resonances.append({
                    "stem": s["name"],
                    "freq_hz": item["freq_hz"],
                    "attenuation_db": item["attenuation_db"]
                })
                
        # Step B: Highpass Filter
        if "hpf" in conf:
            data = fast_biquad_highpass(data, conf["hpf"], sr)
            
        # Step C: Parametric EQ Peaking & Mud Cut
        if "peak_freq" in conf:
            data = fast_biquad_peak(data, conf["peak_freq"], conf["peak_gain"], sr, q=conf.get("peak_q", 1.0))
            
        if "mud_cut_freq" in conf:
            data = fast_biquad_peak(data, conf["mud_cut_freq"], conf["mud_cut_gain"], sr, q=1.2)
            
        # Step D: Sub-Bass Emphasis from Genre Preset
        if t in ["kick", "bass"] and "sub_boost_freq" in preset:
            boost_amt = preset["sub_boost_db"] if t == "bass" else (preset["sub_boost_db"] * 0.8)
            data = fast_biquad_peak(data, preset["sub_boost_freq"], boost_amt, sr, q=1.3)
            
        # Step E: Presence & Air Boost
        if "presence_freq" in conf:
            data = fast_biquad_peak(data, conf["presence_freq"], conf["presence_gain"], sr, q=conf.get("presence_q", 1.0))
            
        if "air_freq" in conf:
            air_gain = conf["air_gain"]
            if t in ["vocal_lead", "vocal_back"]:
                air_gain = air_gain + preset.get("vocal_air_db", 2.0) * 0.5
            data = fast_biquad_highshelf(data, conf["air_freq"], air_gain, sr)
            
        # Step F: Lowpass Filter
        if "lpf" in conf:
            data = fast_biquad_lowpass(data, conf["lpf"], sr)
            
        # Step G: Dynamic Compression
        if "comp_thresh" in conf:
            data = fast_compressor(
                data, sr,
                threshold_db=conf["comp_thresh"],
                ratio=conf.get("comp_ratio", 3.0),
                attack_ms=conf.get("comp_attack_ms", 20),
                release_ms=conf.get("comp_release_ms", 100),
                makeup_db=1.0
            )
            
        # Step H: Sidechain Ducking under Kick (for bass/808)
        if conf.get("sidechain_duck") and kick_data is not None:
            duck_amt = preset.get("sidechain_duck_db", -3.5)
            data = fast_sidechain_duck(data, kick_data, sr, threshold_db=-18.0, duck_db=duck_amt, release_ms=110.0)
            
        # Step I: Spectral Unmasking under Lead Vocals
        if vocal_data is not None and t in ["acoustic_strum", "harp_keys", "strings", "other"]:
            data = apply_spectral_unmasking(data, vocal_data, sr)
            
        # Step J: Vocal Reverb / Space Send
        wet_amount = conf.get("reverb_wet", 0.0)
        if vocal_fx_level is not None and "vocal" in t:
            genre_wet = preset.get("reverb_wet", 0.18)
            wet_amount = (genre_wet * 0.7 + wet_amount * 0.3) * float(vocal_fx_level) * 2.0
        if wet_amount > 0.02:
            data = fast_reverb_send(data, sr, wet=wet_amount)
            
        # Step K: Stereo Width from Genre
        width_val = conf.get("width", 1.0)
        if t in ["vocal_lead", "vocal_back"]:
            width_val *= preset.get("width_vocals", 1.0)
        elif t in ["kick", "bass"]:
            width_val = 0.85
        elif t in ["snare_clap", "percussion_high"]:
            width_val *= preset.get("width_drums", 1.1)
        else:
            width_val *= preset.get("width_instruments", 1.2)
            
        if abs(width_val - 1.0) > 0.05:
            data = fast_stereo_width(data, width_val)
            
        if conf.get("pan", 0.0) != 0.0:
            data = fast_stereo_pan(data, conf["pan"])
            
        # Step L: Channel Fader Gain calibrated by Genre
        gain_db = conf.get("gain_db", 0.0)
        if t == "kick":
            gain_db += preset.get("kick_gain", 0.0)
        elif t == "bass":
            gain_db += preset.get("bass_gain", 0.0)
        elif t in ["vocal_lead", "vocal_back"]:
            gain_db += preset.get("vocal_gain", 0.0)
        elif t in ["snare_clap", "percussion_high"]:
            gain_db += preset.get("snare_gain", 0.0)
        else:
            gain_db += preset.get("other_gain", 0.0)
            
        gain_lin = 10 ** (gain_db / 20.0)
        data = data * gain_lin
        
        summing_bus += data
        
    # 4. Bus Master Calibration: -6.0 dBFS True Headroom
    peak_val = np.max(np.abs(summing_bus))
    target_peak_lin = 10 ** (-6.0 / 20.0) # 0.501 (-6 dBFS)
    if peak_val > 1e-6:
        summing_bus = summing_bus * (target_peak_lin / peak_val)
        
    sf.write(output_path, summing_bus.astype(np.float32), sr, subtype='FLOAT')
    
    # Save raw sum for direct comparison
    raw_path = output_path.replace(".wav", "_raw.wav")
    raw_peak = np.max(np.abs(raw_sum))
    if raw_peak > 1e-6:
        raw_sum = raw_sum * (target_peak_lin / raw_peak)
    sf.write(raw_path, raw_sum.astype(np.float32), sr, subtype='FLOAT')
    
    # Precise metrics
    mix_lufs = calculate_lufs(summing_bus, sr)
    mix_tp = calculate_true_peak(summing_bus, sr)
    mix_dr = calculate_crest_factor(summing_bus)
    
    report = {
        "stems_count": len(stem_paths),
        "genre": genre_key,
        "genre_name": genre_data["name"],
        "subgenre": subgenre_key,
        "subgenre_name": preset["name"],
        "subgenre_desc": preset.get("desc", ""),
        "integrated_lufs": mix_lufs,
        "true_peak_dbfs": mix_tp,
        "dynamic_range_db": mix_dr,
        "headroom": "-6.0 dBFS Peak (32-bit Float Calibrated)",
        "resonances_tamed_count": len(all_tamed_resonances),
        "resonances_tamed": all_tamed_resonances[:8],
        "output_path": output_path,
        "raw_path": raw_path
    }
    
    return output_path, report
