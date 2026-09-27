"""
Midifier Engine: Studio-Grade Audio to MIDI Intelligent Converter
Features:
1. Deep Instrument Classification & General MIDI Program Change embedding
2. Strict Event-Driven Timeline Serialization (Preserving All Rests, Spaces & Exact Durations)
3. Dynamic Velocity Mapping (Human Touch / Playing Feeling from Attack Transients)
4. Advanced Key & Scale Detection with MIDI Key Signature Meta Embedding
5. Advanced BPM & Tempo Detection with MIDI Set Tempo Meta Embedding
6. Polyphonic Chord Progression Recognition & In-Track Chord Marker Injection
7. Primary Neural Transcription (Spotify basic-pitch) with Polyphonic CQT Fallback
"""
import os
import sys
import math
import tempfile
import numpy as np
import soundfile as sf
import scipy.signal as signal

# ==========================================================
# 1. GENERAL MIDI INSTRUMENTS & SPECIFICATIONS
# ==========================================================

GM_INSTRUMENTS = {
    "acoustic_piano": {
        "name": "Acoustic Grand Piano",
        "program": 0,
        "channel": 0,
        "family": "Piano",
        "keywords": ["piano", "grand", "keys", "rhodes", "keyboard", "steinway", "yamaha"]
    },
    "electric_piano": {
        "name": "Electric Piano",
        "program": 4,
        "channel": 0,
        "family": "Keys",
        "keywords": ["electric piano", "epiano", "rhodes", "wurlitzer", "dx7"]
    },
    "acoustic_guitar": {
        "name": "Acoustic Guitar",
        "program": 24,
        "channel": 0,
        "family": "Guitar",
        "keywords": ["acoustic guitar", "guitar", "gtr", "nylon", "steel", "strum", "fingerpicking"]
    },
    "electric_guitar": {
        "name": "Electric Guitar",
        "program": 27,
        "channel": 0,
        "family": "Guitar",
        "keywords": ["electric guitar", "distorted guitar", "clean guitar", "telecaster", "stratocaster", "les paul"]
    },
    "electric_bass": {
        "name": "Electric Bass",
        "program": 33,
        "channel": 0,
        "family": "Bass",
        "keywords": ["bass", "electric bass", "fender bass", "bass guitar", "slap bass"]
    },
    "synth_bass": {
        "name": "Synth Bass / 808",
        "program": 38,
        "channel": 0,
        "family": "Bass",
        "keywords": ["808", "sub", "sub bass", "synth bass", "reese", "moog"]
    },
    "strings": {
        "name": "String Ensemble",
        "program": 48,
        "channel": 0,
        "family": "Strings",
        "keywords": ["strings", "violin", "cello", "viola", "orchestra", "string ensemble", "pad"]
    },
    "brass": {
        "name": "Brass Section",
        "program": 61,
        "channel": 0,
        "family": "Brass",
        "keywords": ["brass", "horns", "trumpet", "trombone", "sax", "saxophone", "brass section"]
    },
    "synth_lead": {
        "name": "Synth Lead",
        "program": 80,
        "channel": 0,
        "family": "Synth",
        "keywords": ["synth", "lead", "saw lead", "square lead", "synth lead", "pluck", "arpeggio"]
    },
    "vocal": {
        "name": "Vocal / Melody",
        "program": 54,
        "channel": 0,
        "family": "Vocal",
        "keywords": ["vocal", "vox", "voice", "acapella", "singing", "lead vox", "choir"]
    },
    "drums": {
        "name": "Drums & Percussion",
        "program": 118,
        "channel": 9,  # MIDI Channel 10 (0-indexed 9)
        "family": "Drums",
        "keywords": ["drum", "drums", "percussion", "kick", "snare", "hihat", "beat", "loop"]
    }
}

KEY_TO_MIDO = {
    "C Major": "C", "A Minor": "Am",
    "G Major": "G", "E Minor": "Em",
    "D Major": "D", "B Minor": "Bm",
    "A Major": "A", "F# Minor": "F#m",
    "E Major": "E", "C# Minor": "C#m",
    "B Major": "B", "G# Minor": "G#m",
    "F# Major": "F#", "D# Minor": "D#m",
    "Gb Major": "F#", "Eb Minor": "Ebm",
    "Db Major": "Db", "Bb Minor": "Bbm",
    "C# Major": "Db", "A# Minor": "Bbm",
    "Ab Major": "Ab", "F Minor": "Fm",
    "Eb Major": "Eb", "C Minor": "Cm",
    "Bb Major": "Bb", "G Minor": "Gm",
    "F Major": "F", "D Minor": "Dm",
}

CHORD_TEMPLATES = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "sus4": {0, 5, 7},
    "sus2": {0, 2, 7},
    "5": {0, 7},
}

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# ==========================================================
# 2. INSTRUMENT CLASSIFICATION
# ==========================================================

def classify_instrument(audio_path, y, sr, notes):
    """
    Classifies the audio into an appropriate General MIDI instrument and program.
    Combines filename hints with spectral centroid, zero-crossing rate, pitch range, and polyphony.
    """
    path_lower = os.path.basename(audio_path).lower()

    # 1. Filename heuristic checks
    for key, spec in GM_INSTRUMENTS.items():
        for kw in spec["keywords"]:
            if kw in path_lower:
                return spec

    # 2. Audio Signal Acoustic Characterization
    if y is None or len(y) == 0:
        return GM_INSTRUMENTS["acoustic_piano"]

    mono = y if y.ndim == 1 else np.mean(y, axis=0)

    # Spectral Centroid
    freqs, psd = signal.welch(mono, fs=sr, nperseg=min(2048, len(mono)))
    total_power = np.sum(psd) + 1e-12
    centroid = float(np.sum(freqs * psd) / total_power)

    # Zero Crossing Rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(mono)))) / 2.0)

    # Pitch statistics from transcribed notes
    if notes and len(notes) > 0:
        pitches = [n["pitch"] for n in notes]
        median_pitch = float(np.median(pitches))
        min_pitch = float(np.min(pitches))
        max_pitch = float(np.max(pitches))
    else:
        median_pitch = 60.0
        min_pitch = 48.0
        max_pitch = 72.0

    # Polyphony density calculation (average notes sounding simultaneously)
    avg_polyphony = 1.0
    if notes and len(notes) > 1:
        time_points = np.linspace(notes[0]["start"], notes[-1]["end"], min(100, len(notes) * 2))
        concurrent_counts = [
            sum(1 for n in notes if n["start"] <= t <= n["end"])
            for t in time_points
        ]
        if concurrent_counts:
            avg_polyphony = float(np.mean(concurrent_counts))

    # Acoustic Decision Tree
    # Sub / Electric Bass
    if median_pitch < 48 or (centroid < 850 and max_pitch < 58):
        if centroid < 550:
            return GM_INSTRUMENTS["synth_bass"]
        return GM_INSTRUMENTS["electric_bass"]

    # Drums / Percussion
    if zcr > 0.16 and centroid > 3800 and avg_polyphony <= 1.2:
        return GM_INSTRUMENTS["drums"]

    # High Lead / Synth / Flute
    if median_pitch > 74 and centroid > 3200:
        return GM_INSTRUMENTS["synth_lead"]

    # Polyphonic Keyboard / Guitar / Strings
    if avg_polyphony >= 2.0:
        if centroid < 1800:
            return GM_INSTRUMENTS["acoustic_piano"]
        elif centroid > 2600:
            return GM_INSTRUMENTS["acoustic_guitar"]
        else:
            return GM_INSTRUMENTS["strings"]

    # Monophonic Vocals or Solo Lead
    if avg_polyphony < 1.4:
        if 48 <= median_pitch <= 76 and 900 <= centroid <= 2800:
            return GM_INSTRUMENTS["vocal"]
        return GM_INSTRUMENTS["synth_lead"]

    # Default versatile studio piano
    return GM_INSTRUMENTS["acoustic_piano"]

# ==========================================================
# 3. ADVANCED KEY & TEMPO DETECTION
# ==========================================================

def detect_musical_key(y, sr):
    """
    Detects the musical key using Essentia if available, falling back to CENS Krumhansl-Schmuckler.
    """
    try:
        from generalstems import detect_key_advanced
        key = detect_key_advanced(y, sr)
        if key and key.strip():
            return key.strip()
    except Exception:
        pass

    try:
        import librosa
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, n_chroma=12)
        mean_chroma = np.mean(chroma, axis=1)
        norm = np.linalg.norm(mean_chroma)
        if norm > 1e-6:
            mean_chroma = mean_chroma / norm

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

        best_corr = -1
        best_key = "C"
        best_mode = "Major"

        for i, k in enumerate(keys):
            maj_p = np.roll(major_profile, i)
            min_p = np.roll(minor_profile, i)
            c_maj = np.corrcoef(mean_chroma, maj_p / np.linalg.norm(maj_p))[0, 1]
            c_min = np.corrcoef(mean_chroma, min_p / np.linalg.norm(min_p))[0, 1]

            if c_maj > best_corr:
                best_corr = c_maj
                best_key = k
                best_mode = "Major"
            if c_min > best_corr:
                best_corr = c_min
                best_key = k
                best_mode = "Minor"

        return f"{best_key} {best_mode}"
    except Exception:
        return "C Major"

def detect_musical_bpm(y, sr):
    """
    Detects BPM snapped to nearest whole musical integer.
    """
    try:
        from generalstems import detect_tempo_advanced
        bpm = detect_tempo_advanced(y, sr)
        bpm_int = int(round(float(bpm)))
        if 40 <= bpm_int <= 240:
            return bpm_int
    except Exception:
        pass

    try:
        import librosa
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo_val = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
        tempo_int = int(round(tempo_val))
        if 50 <= tempo_int <= 220:
            return tempo_int
    except Exception:
        pass

    return 120

# ==========================================================
# 4. CHORD PROGRESSION RECOGNITION
# ==========================================================

def detect_chord_progression(notes, total_duration, step_sec=0.25):
    """
    Scans the transcribed notes over time slices, identifying root note and triad/seventh chord qualities.
    Returns:
    - chord_markers: list of {'time': float, 'chord': str, 'duration': float}
    - unique_progression: list of distinct chord names (e.g. ['Am', 'F', 'C', 'G'])
    """
    if not notes or len(notes) < 2:
        return [], []

    n_steps = max(1, int(total_duration / step_sec))
    detected_slices = []

    for s in range(n_steps):
        t = s * step_sec
        active = [n["pitch"] for n in notes if n["start"] <= t < n["end"]]
        pcs = sorted(list(set(p % 12 for p in active)))

        if len(pcs) >= 2:
            best_chord = None
            best_score = -1

            # Determine lowest note as probable bass / root hint
            bass_pc = min(active) % 12

            for root_idx in range(12):
                rel_pcs = set((pc - root_idx) % 12 for pc in pcs)
                for quality, template in CHORD_TEMPLATES.items():
                    intersection = len(rel_pcs & template)
                    extra = len(rel_pcs - template)
                    missing = len(template - rel_pcs)
                    
                    # Boost score if candidate root matches the lowest bass note
                    bass_bonus = 1 if root_idx == bass_pc else 0
                    score = (intersection * 3) - extra - (missing * 2) + bass_bonus

                    if score > best_score and intersection >= 2:
                        best_score = score
                        suffix = "" if quality == "maj" else ("m" if quality == "min" else quality)
                        best_chord = f"{NOTE_NAMES[root_idx]}{suffix}"

            if best_chord:
                detected_slices.append((t, best_chord))

    if not detected_slices:
        return [], []

    # Group consecutive identical slices into sustained chords (minimum 0.3s)
    chord_markers = []
    current_chord = detected_slices[0][1]
    chord_start = detected_slices[0][0]
    chord_count = 1

    for t, ch in detected_slices[1:]:
        if ch == current_chord:
            chord_count += 1
        else:
            chord_dur = chord_count * step_sec
            if chord_dur >= 0.30:
                chord_markers.append({
                    "time": round(chord_start, 2),
                    "chord": current_chord,
                    "duration": round(chord_dur, 2)
                })
            current_chord = ch
            chord_start = t
            chord_count = 1

    # Append trailing chord
    if chord_count * step_sec >= 0.30:
        chord_markers.append({
            "time": round(chord_start, 2),
            "chord": current_chord,
            "duration": round(chord_count * step_sec, 2)
        })

    # Build unique sequence of chords (e.g. ['Am', 'F', 'C', 'G'])
    unique_progression = []
    for cm in chord_markers:
        ch = cm["chord"]
        if not unique_progression or unique_progression[-1] != ch:
            unique_progression.append(ch)

    return chord_markers, unique_progression

# ==========================================================
# 5. HUMAN FEELING & VELOCITY DYNAMICS MAPPING
# ==========================================================

def calculate_human_velocities(y, sr, notes):
    """
    Calculates dynamic MIDI velocities for each note based on the audio attack RMS energy,
    ensuring natural touch, accents, and ghost notes instead of static robotic volume.
    """
    if y is None or len(y) == 0 or not notes:
        for n in notes:
            n["velocity"] = 80
        return notes

    mono = y if y.ndim == 1 else np.mean(y, axis=0)
    audio_len = len(mono)

    # Track-wide RMS reference
    track_peak_rms = float(np.percentile(np.abs(mono), 95)) + 1e-6

    for n in notes:
        st_sec = n["start"]
        en_sec = n["end"]

        # Attack transient window: 80ms to 120ms from start of note
        st_sample = max(0, int(st_sec * sr))
        attack_window = int(min(en_sec - st_sec, 0.12) * sr)
        en_sample = min(audio_len, st_sample + max(32, attack_window))

        slice_audio = mono[st_sample:en_sample]
        if len(slice_audio) > 0:
            attack_rms = float(np.sqrt(np.mean(slice_audio**2)))
        else:
            attack_rms = 0.05

        # Decibel ratio relative to track peak
        db = 20.0 * np.log10(max(1e-5, attack_rms) / track_peak_rms)
        # Dynamic range: -34 dB (ghost note) to 0 dB (full fortissimo accent)
        clamped_db = max(-34.0, min(0.0, db))
        ratio = (clamped_db + 34.0) / 34.0  # 0.0 to 1.0

        # Map to expressive MIDI velocity range: 35 (pianissimo / ghost) to 126 (fortissimo)
        base_vel = 35 + (ratio * 91.0)

        # Blend with neural amplitude if available from basic-pitch
        if "amplitude" in n and n["amplitude"] is not None:
            model_amp = max(0.0, min(1.0, float(n["amplitude"])))
            model_vel = 35 + (model_amp * 91.0)
            final_vel = int(round((0.55 * base_vel) + (0.45 * model_vel)))
        else:
            final_vel = int(round(base_vel))

        n["velocity"] = max(28, min(127, final_vel))

    return notes

# ==========================================================
# 6. ABSOLUTE EVENT-DRIVEN MIDI SERIALIZATION
# ==========================================================

def serialize_to_midi_file(notes, chord_markers, instrument_spec, key_name, bpm, output_midi_path):
    """
    Serializes notes and chord markers into a standard Type 0/1 MIDI file.
    Critically: calculates exact delta time in ticks for every event so that:
    - ALL rests / spaces between notes and phrases are 100% preserved.
    - Note durations are 100% exact.
    - Instrument Program Change is set.
    - Track Name, Set Tempo, Time Signature, and Key Signature meta events are set.
    """
    import mido
    from mido import Message, MetaMessage, MidiFile, MidiTrack

    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480
    ticks_per_second = (ticks_per_beat * bpm) / 60.0
    channel = instrument_spec.get("channel", 0)
    program = instrument_spec.get("program", 0)
    inst_name = instrument_spec.get("name", "Acoustic Grand Piano")

    # 1. Header Meta Events at Time 0
    track.append(MetaMessage('track_name', name=inst_name, time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    # Safe Key Signature mapping
    mido_key = KEY_TO_MIDO.get(key_name, "C")
    try:
        track.append(MetaMessage('key_signature', key=mido_key, time=0))
    except Exception as ke:
        print(f"[MIDIFIER] Warning setting key_signature '{mido_key}': {ke}")

    # Program Change for Instrument
    track.append(Message('program_change', program=program, channel=channel, time=0))

    # 2. Build Event List on Absolute Timeline
    events = []

    # Note Events
    for n in notes:
        st = max(0.0, float(n["start"]))
        en = max(st + 0.02, float(n["end"]))
        pitch = max(0, min(127, int(round(n["pitch"]))))
        vel = max(1, min(127, int(round(n.get("velocity", 80)))))

        events.append((st, 'note_on', pitch, vel))
        events.append((en, 'note_off', pitch, 64))

    # Chord Markers
    if chord_markers:
        for cm in chord_markers:
            t = max(0.0, float(cm["time"]))
            events.append((t, 'text', f"[CHORD: {cm['chord']}]", 0))

    # 3. Sort Events Chronologically
    # Tie-breaking priority at same millisecond:
    # note_off (0) < text marker (1) < note_on (2)
    # This prevents voice-stealing bugs in DAWs
    def event_sort_key(e):
        t = e[0]
        etype = e[1]
        priority = 0 if etype == 'note_off' else (1 if etype == 'text' else 2)
        return (t, priority)

    events.sort(key=event_sort_key)

    # 4. Serialize with Microsecond-Exact Delta-Ticks
    current_time_sec = 0.0

    for evt_time_sec, evt_type, val1, val2 in events:
        delta_sec = max(0.0, evt_time_sec - current_time_sec)
        delta_ticks = int(round(delta_sec * ticks_per_second))

        if evt_type == 'note_on':
            track.append(Message('note_on', note=val1, velocity=val2, channel=channel, time=delta_ticks))
        elif evt_type == 'note_off':
            track.append(Message('note_off', note=val1, velocity=val2, channel=channel, time=delta_ticks))
        elif evt_type == 'text':
            track.append(MetaMessage('text', text=val1, time=delta_ticks))

        current_time_sec = evt_time_sec

    # End of Track Marker with 1-beat trailing headroom
    trailing_ticks = int(round(0.5 * ticks_per_second))
    track.append(MetaMessage('end_of_track', time=trailing_ticks))

    # Save to destination
    mid.save(output_midi_path)
    return output_midi_path

# ==========================================================
# 7. CORE TRANSCRIPTION ENGINES
# ==========================================================

def transcribe_with_basic_pitch(audio_path, note_sensitivity=0.5, min_note_length=58):
    """
    Transcribes audio using Spotify's state-of-the-art basic-pitch model.
    Returns list of dicts: {'start': float, 'end': float, 'pitch': int, 'amplitude': float}
    """
    from basic_pitch.inference import predict
    
    # In basic-pitch:
    # onset_threshold corresponds to note sensitivity (higher sensitivity = lower onset_threshold)
    onset_thresh = max(0.1, min(0.9, 1.0 - (float(note_sensitivity) * 0.75)))
    frame_thresh = max(0.08, onset_thresh * 0.75)
    min_len_ms = max(20.0, float(min_note_length))

    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=onset_thresh,
        frame_threshold=frame_thresh,
        minimum_note_length=min_len_ms,
        minimum_frequency=30.0,
        maximum_frequency=3500.0
    )

    notes = []
    # note_events format: (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bends)
    for ev in note_events:
        st = float(ev[0])
        en = float(ev[1])
        pitch = int(round(ev[2]))
        amp = float(ev[3]) if len(ev) > 3 else 0.8
        if en > st and 12 <= pitch <= 127:
            notes.append({
                "start": st,
                "end": en,
                "pitch": pitch,
                "amplitude": amp
            })

    # Sort notes by start time
    notes.sort(key=lambda n: n["start"])
    return notes

def transcribe_with_polyphonic_cqt(y, sr, note_sensitivity=0.5, min_note_length=58):
    """
    Studio-grade polyphonic CQT fallback engine.
    Extracts harmonic peaks at onset transients across 5 octaves (C2 to C7)
    and tracks note durations accurately.
    """
    import librosa

    hop_length = 512
    frame_duration = hop_length / float(sr)
    min_dur_sec = max(0.025, float(min_note_length) / 1000.0)

    # 1. Harmonic extraction
    y_harm = librosa.effects.harmonic(y, margin=3.0)

    # 2. Constant-Q Transform: 60 bins (5 octaves), 12 bins per octave starting at C2 (~65.4 Hz)
    fmin = librosa.note_to_hz('C2')
    cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=60, bins_per_octave=12))

    # Convert to dB normalized
    ref_val = np.max(cqt) + 1e-9
    cqt_db = librosa.amplitude_to_db(cqt, ref=ref_val)

    # 3. Onset Detection
    onset_env = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=hop_length)
    delta_sens = max(0.04, min(0.35, 0.40 - (float(note_sensitivity) * 0.32)))
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        delta=delta_sens
    )

    if len(onset_frames) == 0:
        return []

    notes = []
    n_frames = cqt_db.shape[1]
    db_thresh = -28.0 + (float(note_sensitivity) * 12.0)  # -28 dB to -16 dB

    for i, oframe in enumerate(onset_frames):
        st_sec = oframe * frame_duration
        next_onset_frame = onset_frames[i + 1] if (i + 1) < len(onset_frames) else n_frames

        # Spectral slice at onset
        spec_slice = cqt_db[:, oframe]
        peaks, props = signal.find_peaks(spec_slice, height=db_thresh, distance=2, prominence=4.0)

        for p_idx in peaks:
            pitch_midi = int(round(librosa.hz_to_midi(fmin * (2 ** (p_idx / 12.0)))))
            if not (20 <= pitch_midi <= 108):
                continue

            # Trace peak forward in time until energy drops by >14 dB or next onset
            onset_amp_db = spec_slice[p_idx]
            end_frame = oframe + 1

            while end_frame < min(next_onset_frame, n_frames):
                cur_val = cqt_db[p_idx, end_frame]
                if cur_val < (onset_amp_db - 14.0) or cur_val < db_thresh:
                    break
                end_frame += 1

            en_sec = end_frame * frame_duration
            dur = en_sec - st_sec

            if dur >= min_dur_sec:
                # Normalized linear amplitude
                lin_amp = float(props["peak_heights"][list(peaks).index(p_idx)] + 40.0) / 40.0
                notes.append({
                    "start": round(st_sec, 3),
                    "end": round(en_sec, 3),
                    "pitch": pitch_midi,
                    "amplitude": max(0.1, min(1.0, lin_amp))
                })

    notes.sort(key=lambda n: n["start"])
    return notes

# ==========================================================
# 8. MASTER CONVERT PIPELINE
# ==========================================================

def convert_audio_to_midi(audio_path, output_midi_path=None, note_sensitivity=0.5, min_note_length=58):
    """
    Main entrypoint: converts an audio track to professional, studio-ready MIDI.
    Detects instrument, BPM, Key, Chords, and Velocity Feeling, preserving all rests and spaces.
    """
    if not output_midi_path:
        base = os.path.splitext(audio_path)[0]
        output_midi_path = f"{base}.mid"

    os.makedirs(os.path.dirname(output_midi_path) or ".", exist_ok=True)

    # 1. Load Audio cleanly for acoustic & harmonic analysis
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    total_duration = float(len(y) / sr)

    # 2. Musical Key & BPM Detection
    detected_key = detect_musical_key(y, sr)
    detected_bpm = detect_musical_bpm(y, sr)

    # 3. Note Transcription (Try Spotify basic-pitch, fallback to CQT)
    notes = []
    engine_name = "Studio CQT Polyphonic"

    try:
        notes = transcribe_with_basic_pitch(
            audio_path,
            note_sensitivity=note_sensitivity,
            min_note_length=min_note_length
        )
        if notes:
            engine_name = "Spotify basic-pitch Neural"
    except Exception as bp_err:
        print(f"[MIDIFIER] Notice: basic-pitch not available ({bp_err}), using studio CQT engine...")
        notes = []

    if not notes:
        engine_name = "Studio CQT Polyphonic"
        try:
            notes = transcribe_with_polyphonic_cqt(
                y, sr,
                note_sensitivity=note_sensitivity,
                min_note_length=min_note_length
            )
        except Exception as cqt_err:
            print(f"[MIDIFIER] CQT error: {cqt_err}")
            notes = []

    if not notes:
        raise RuntimeError("No musical notes could be extracted from audio. Verify that the file contains audible melodic or harmonic content.")

    # 4. Human Dynamic Velocity / Feeling Calculation
    notes = calculate_human_velocities(y, sr, notes)

    # 5. Instrument Classification & GM Program Change
    instrument_spec = classify_instrument(audio_path, y, sr, notes)

    # 6. Chord Recognition
    chord_markers, chord_progression = detect_chord_progression(notes, total_duration)

    # 7. Serialize to MIDI with Strict Delta-Time Rest Preservation
    serialize_to_midi_file(
        notes=notes,
        chord_markers=chord_markers,
        instrument_spec=instrument_spec,
        key_name=detected_key,
        bpm=detected_bpm,
        output_midi_path=output_midi_path
    )

    # 8. Compile Comprehensive Metadata Report
    vels = [n["velocity"] for n in notes]
    min_vel = int(min(vels)) if vels else 64
    max_vel = int(max(vels)) if vels else 96

    # Determine polyphony state
    overlapping = False
    for i in range(len(notes) - 1):
        if notes[i]["end"] > notes[i + 1]["start"] + 0.05:
            overlapping = True
            break
    poly_label = "Polyphonic (Chords & Harmonies)" if overlapping else "Monophonic (Melodic Lead)"

    # Compact notes preview for instant browser playback
    preview_notes = [
        {"s": round(float(n["start"]), 3), "e": round(float(n["end"]), 3), "p": int(n["pitch"]), "v": int(n["velocity"])}
        for n in notes[:350]
    ]

    metadata = {
        "status": "success",
        "instrument": instrument_spec["name"],
        "gm_program": instrument_spec["program"],
        "channel": instrument_spec["channel"],
        "family": instrument_spec["family"],
        "key": detected_key,
        "bpm": detected_bpm,
        "chord_progression": chord_progression[:8] if chord_progression else ["Single Melody"],
        "note_count": len(notes),
        "polyphony": poly_label,
        "velocity_range": f"{min_vel} - {max_vel} (Dynamic Touch)",
        "engine": engine_name,
        "duration_sec": round(total_duration, 2),
        "notes_preview": preview_notes
    }

    return output_midi_path, metadata
