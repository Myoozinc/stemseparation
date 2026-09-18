"""
Myooz Audio Intelligence Suite - Unified API
Modules:
1. /process: Demucs Stem Separation + BPM/Key
2. /create_download_zip: Stems batch exporter
3. /process_midi: Audio to MIDI Converter (basic-pitch)
4. /process_samples: Sample Slicer & Kit Packager
5. /process_mix: Multi-Stem AI Mixing Studio (pedalboard)
6. /process_master: AI Mastering Studio (LUFS / true-peak limiter)
"""
import os
# Strictly enforce 2-thread limit for cloud container cgroups
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["TORCH_NUM_THREADS"] = "2"

import json
import gradio as gr
from midifier_engine import convert_audio_to_midi
from sampler_engine import slice_audio_samples
from mixter_engine import process_and_mix_stems
from master_engine import master_audio

# --- Existing Demucs / Stem separation functions ---
try:
    from generalstems import (
        mp3_to_wav,
        demucs_separate,
        refine_drums,
        detect_key_advanced,
        detect_tempo_advanced
    )
except ImportError:
    # Local placeholder if generalstems is not present locally
    def mp3_to_wav(p, o): return p
    def demucs_separate(p, o): return {}, ""
    def refine_drums(p, o): return {}
    def detect_key_advanced(p): return "C Major"
    def detect_tempo_advanced(p): return "120"

def extract_file_info(f):
    """Safely extracts (file_path, original_filename) from any Gradio file object, dictionary or URL"""
    if isinstance(f, dict):
        path = f.get('path') or f.get('name') or ''
        orig_name = f.get('orig_name') or (os.path.basename(path) if path else 'audio.wav')
    elif hasattr(f, 'name'):
        path = f.name
        orig_name = getattr(f, 'orig_name', os.path.basename(path))
    else:
        path = str(f)
        orig_name = os.path.basename(path)

    # If path is a URL referencing a file on this server in /tmp/, map it directly to local disk
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
        if "/tmp/" in path:
            local_candidate = path[path.find("/tmp/"):]
            if os.path.exists(local_candidate):
                return local_candidate, orig_name
        try:
            import urllib.request, time
            clean_name = os.path.basename(path.split("?")[0]) or orig_name
            local_dl = f"/tmp/dl_{int(time.time()*1000)}_{clean_name}"
            urllib.request.urlretrieve(path, local_dl)
            if os.path.exists(local_dl):
                return local_dl, clean_name
        except Exception as dl_err:
            print(f"[WARN] Failed downloading remote audio URL {path}: {dl_err}")

    return path, orig_name

def process_stems(song_file, options, auth_token=""):
    """Original stem separation endpoint"""
    if not song_file:
        return [None] * 8 + ["NO_FILE"]
    try:
        out_dir = "/tmp/stems_out"
        os.makedirs(out_dir, exist_ok=True)
        song_path, orig_name = extract_file_info(song_file)
        if not song_path or not os.path.exists(song_path):
            return [None] * 8 + [f"PROCESSING_ERROR: File not accessible on server: {song_file}"]
            
        wav_path = os.path.join(out_dir, "input_song.wav")
        wav_path = mp3_to_wav(song_path, wav_path)
        stems = {}
        drum_refined = {}

        if any(o in options for o in ["Vocals", "Bass", "Other", "Drums", "All"]):
            stems, _ = demucs_separate(wav_path, out_dir)

        if ("Refine Drums" in options or "All" in options) and "drums" in stems:
            drum_refined = refine_drums(stems["drums"], out_dir)

        key = "N/A"
        tempo = "N/A"
        if "Info" in options or "All" in options:
            try:
                import librosa
                # If drums stem was separated, use it for 100% pure rhythm detection with zero melodic bleed
                tempo_src = stems.get("drums") if (stems.get("drums") and os.path.exists(stems.get("drums"))) else wav_path
                y_tempo, sr_t = librosa.load(tempo_src, mono=True, sr=22050, duration=30.0)
                tempo = detect_tempo_advanced(y_tempo, sr_t)
                
                # Load first 25 seconds for key detection
                y_key, sr_k = librosa.load(wav_path, mono=True, sr=22050, duration=25.0)
                key = detect_key_advanced(y_key, sr_k)
            except Exception as kerr:
                print(f"[WARN] Key/tempo detection: {kerr}")
                key = "N/A"
                tempo = "N/A"

        info_msg = f"Key: {key} | Tempo: {tempo} BPM | SUCCESS:99:99"

        return (
            wav_path,
            stems.get("vocals"),
            stems.get("bass"),
            stems.get("other"),
            stems.get("drums"),
            drum_refined.get("kick"),
            drum_refined.get("snare"),
            drum_refined.get("hihat"),
            info_msg
        )
    except Exception as e:
        print(f"[ERROR] Stem processing: {e}")
        return [None] * 8 + [f"PROCESSING_ERROR: {e}"]

# --- Midifier Endpoint ---
def process_midi(audio_file, note_sensitivity=0.5, min_note_length=58):
    if not audio_file:
        return None, "No audio file provided."
    try:
        midi_path, info = convert_audio_to_midi(
            audio_file,
            note_sensitivity=note_sensitivity,
            min_note_length=min_note_length
        )
        return midi_path, f"SUCCESS: {info}"
    except Exception as e:
        return None, f"ERROR: {e}"

# --- Sampler Endpoint ---
def process_samples(audio_file, mode="transients", max_samples=16):
    if not audio_file:
        return None, "No audio file provided.", []
    try:
        zip_path, sample_list = slice_audio_samples(
            audio_file,
            mode=mode,
            max_samples=int(max_samples)
        )
        preview_paths = [s["path"] for s in sample_list[:8]]
        # Pad previews to 8 slots
        while len(preview_paths) < 8:
            preview_paths.append(None)

        msg = f"SUCCESS: Extracted {len(sample_list)} samples into {os.path.basename(zip_path)}"
        return [zip_path, msg] + preview_paths
    except Exception as e:
        return [None, f"ERROR: {e}"] + [None] * 8

# --- Mixter Endpoint ---
def process_mix(files, mix_style="urbano", vocal_fx=0.3, subgenre="neo_perreo", *args, **kwargs):
    if not files:
        return None, "No stem files provided."
    try:
        stem_items = []
        for f in files:
            p, orig_name = extract_file_info(f)
            if p and os.path.exists(p):
                stem_items.append((p, orig_name))
        
        if not stem_items:
            return None, "No valid stem audio files could be accessed on server."
            
        # Support colon syntax "genre:subgenre"
        if ":" in str(mix_style):
            parts = str(mix_style).split(":", 1)
            mix_style = parts[0]
            subgenre = parts[1]

        out_wav, report = process_and_mix_stems(
            stem_items,
            mix_style=mix_style,
            subgenre=subgenre,
            vocal_fx_level=float(vocal_fx)
        )
        report_json = json.dumps(report)
        report_str = f"SUCCESS:{report_json}"
        return out_wav, report_str
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"ERROR: {e}"

# --- Master Endpoint ---
def process_master(audio_file, genre="urbano", style="club_banger", warmth=None, stereo_spread=None, air=None):
    if not audio_file:
        return None, "No mix audio provided."
    try:
        path, _ = extract_file_info(audio_file)
        if not path or not os.path.exists(path):
            return None, f"Audio file not found on server: {path}"
            
        # Backward compatibility with older target_profile parameters
        if genre in ["streaming", "club", "dynamic"]:
            style = "club_banger" if genre == "club" else ("streaming" if genre == "streaming" else "analog_warmth")
            genre = "urbano"

        out_master, metrics = master_audio(
            path,
            genre=genre,
            style=style
        )
        report_json = json.dumps(metrics)
        report_str = f"SUCCESS:{report_json}"
        return out_master, report_str
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"ERROR: {e}"

# --- Gradio Application with defined API Names ---
with gr.Blocks(title="Myooz Audio Intelligence Suite") as app:
    gr.Markdown("# Myooz Audio Intelligence Suite API")

    # Stemer
    with gr.Tab("Stemer"):
        song_in = gr.Audio(type="filepath", label="Song")
        opts = gr.CheckboxGroup(choices=["All", "Vocals", "Bass", "Other", "Drums", "Refine Drums", "Info"], value=["All"])
        auth_tok = gr.Textbox(label="Auth", visible=False)
        stemer_btn = gr.Button("Separate")
        out_wav = gr.Audio(label="WAV")
        out_vox = gr.Audio(label="Vocals")
        out_bass = gr.Audio(label="Bass")
        out_other = gr.Audio(label="Other")
        out_drums = gr.Audio(label="Drums")
        out_kick = gr.Audio(label="Kick")
        out_snare = gr.Audio(label="Snare")
        out_hihat = gr.Audio(label="Hihat")
        out_info = gr.Textbox(label="Status")
        stemer_btn.click(
            fn=process_stems,
            inputs=[song_in, opts, auth_tok],
            outputs=[out_wav, out_vox, out_bass, out_other, out_drums, out_kick, out_snare, out_hihat, out_info],
            api_name="process"
        )

    # Midifier
    with gr.Tab("Midifier"):
        midi_audio_in = gr.Audio(type="filepath", label="Audio to Transcribe")
        midi_sens = gr.Slider(0.1, 0.9, value=0.5, label="Sensitivity")
        midi_min_len = gr.Slider(20, 200, value=58, label="Min Note Length (ms)")
        midifier_btn = gr.Button("Convert to MIDI")
        midi_out_file = gr.File(label="Result MIDI")
        midi_status = gr.Textbox(label="Status")
        midifier_btn.click(
            fn=process_midi,
            inputs=[midi_audio_in, midi_sens, midi_min_len],
            outputs=[midi_out_file, midi_status],
            api_name="process_midi"
        )

    # Sampler
    with gr.Tab("Sampler"):
        sampler_audio_in = gr.Audio(type="filepath", label="Audio to Slice")
        sampler_mode = gr.Radio(choices=["transients", "beats"], value="transients", label="Slicing Mode")
        sampler_max = gr.Slider(4, 32, value=16, step=1, label="Max Samples")
        sampler_btn = gr.Button("Slice Samples")
        sampler_zip = gr.File(label="Download ZIP Kit")
        sampler_status = gr.Textbox(label="Status")
        p0 = gr.Audio(label="Pad 1")
        p1 = gr.Audio(label="Pad 2")
        p2 = gr.Audio(label="Pad 3")
        p3 = gr.Audio(label="Pad 4")
        p4 = gr.Audio(label="Pad 5")
        p5 = gr.Audio(label="Pad 6")
        p6 = gr.Audio(label="Pad 7")
        p7 = gr.Audio(label="Pad 8")
        sampler_btn.click(
            fn=process_samples,
            inputs=[sampler_audio_in, sampler_mode, sampler_max],
            outputs=[sampler_zip, sampler_status, p0, p1, p2, p3, p4, p5, p6, p7],
            api_name="process_samples"
        )

    # Mixter
    with gr.Tab("Mixter"):
        mix_files = gr.File(file_count="multiple", label="Upload Stems")
        mix_genre = gr.Dropdown(
            choices=["urbano", "electronic", "pop", "hiphop", "rock", "acoustic"],
            value="urbano",
            label="Genre"
        )
        mix_vocal_fx = gr.Slider(0.0, 1.0, value=0.3, label="Vocal Space / Reverb")
        mix_subgenre = gr.Textbox(value="neo_perreo", label="Subgenre Preset")
        mix_btn = gr.Button("AI Mix Stems")
        mix_out_wav = gr.Audio(label="Final Mixdown WAV")
        mix_status = gr.Textbox(label="Status / Metrics")
        mix_btn.click(
            fn=process_mix,
            inputs=[mix_files, mix_genre, mix_vocal_fx, mix_subgenre],
            outputs=[mix_out_wav, mix_status],
            api_name="process_mix"
        )

    # Master
    with gr.Tab("Master"):
        master_audio_in = gr.Audio(type="filepath", label="Stereo Mixdown")
        master_genre = gr.Dropdown(
            choices=["urbano", "pop", "electronic", "hiphop", "rock", "acoustic"],
            value="urbano",
            label="Genre"
        )
        master_style = gr.Textbox(value="club_banger", label="Style Preset")
        master_btn = gr.Button("Master Audio")
        master_out_wav = gr.Audio(label="Mastered Audio")
        master_status = gr.Textbox(label="Status / Metrics")
        master_btn.click(
            fn=process_master,
            inputs=[master_audio_in, master_genre, master_style],
            outputs=[master_out_wav, master_status],
            api_name="process_master"
        )

# Background Demucs model prewarm to eliminate first-request wait time
def _prewarm_demucs():
    try:
        import torch
        from generalstems import get_cached_demucs_model
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        get_cached_demucs_model(dev)
        print("[PREWARM] Demucs model cached successfully in memory.")
    except Exception as e:
        print(f"[PREWARM] Notice: {e}")

import threading
threading.Thread(target=_prewarm_demucs, daemon=True).start()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
