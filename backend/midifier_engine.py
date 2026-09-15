"""
Midifier Engine: Audio to MIDI conversion
Uses Spotify's basic-pitch model with fallback to librosa pitch tracking.
Generates standard .mid files with polyphonic and monophonic support.
"""
import os
import tempfile
import numpy as np
import soundfile as sf

def convert_audio_to_midi(audio_path, output_midi_path=None, note_sensitivity=0.5, min_note_length=58):
    """
    Converts audio file to MIDI.
    Args:
        audio_path (str): Path to input audio (.wav, .mp3)
        output_midi_path (str, optional): Target .mid path
        note_sensitivity (float): Onset threshold (0.1 to 0.9)
        min_note_length (int): Minimum note length in ms
    Returns:
        tuple: (output_midi_path, notes_count_info)
    """
    if not output_midi_path:
        base = os.path.splitext(audio_path)[0]
        output_midi_path = f"{base}.mid"

    try:
        # 1. Try Spotify basic-pitch (State of the art)
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH

        output_dir = os.path.dirname(output_midi_path) or "."
        save_midi = True
        sonify_midi = False
        save_model_outputs = False
        save_notes = False

        predict_and_save(
            [audio_path],
            output_dir,
            save_midi=save_midi,
            sonify_midi=sonify_midi,
            save_model_outputs=save_model_outputs,
            save_notes=save_notes,
            model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=float(note_sensitivity),
            minimum_note_length=float(min_note_length)
        )

        expected_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_basic_pitch.mid")
        if os.path.exists(expected_file):
            if expected_file != output_midi_path:
                os.replace(expected_file, output_midi_path)
            return output_midi_path, "Converse completed with basic-pitch model."

    except ImportError:
        pass
    except Exception as e:
        print(f"[MIDIFIER] basic-pitch error: {e}, attempting librosa fallback...")

    # 2. Fallback: Librosa + Mido pitch tracking
    try:
        import librosa
        import mido
        from mido import Message, MidiFile, MidiTrack

        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        hop_length = 512
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length
        )

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

        frame_duration = hop_length / sr
        current_midi_note = None
        note_start_time = 0
        ticks_per_second = 960  # standard midi resolution

        for frame_idx, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
            time_now = frame_idx * frame_duration
            if voiced and not np.isnan(freq):
                midi_val = int(round(librosa.hz_to_midi(freq)))
                midi_val = max(0, min(127, midi_val))
                if current_midi_note is None:
                    current_midi_note = midi_val
                    note_start_time = time_now
                elif midi_val != current_midi_note:
                    # Note off previous
                    duration = time_now - note_start_time
                    ticks = int(duration * ticks_per_second)
                    track.append(Message('note_on', note=current_midi_note, velocity=85, time=0))
                    track.append(Message('note_off', note=current_midi_note, velocity=64, time=ticks))
                    current_midi_note = midi_val
                    note_start_time = time_now
            else:
                if current_midi_note is not None:
                    duration = time_now - note_start_time
                    ticks = int(duration * ticks_per_second)
                    track.append(Message('note_on', note=current_midi_note, velocity=85, time=0))
                    track.append(Message('note_off', note=current_midi_note, velocity=64, time=ticks))
                    current_midi_note = None

        mid.save(output_midi_path)
        return output_midi_path, "Converse completed with harmonic tracking."

    except Exception as e:
        raise RuntimeError(f"Failed to convert audio to MIDI: {e}")
