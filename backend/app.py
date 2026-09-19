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
from generative_engine import generate_instrument_stem

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

        # Normalize options
        if not options:
            options = ["All"]
        elif isinstance(options, str):
            options = [options]

        if any(o in options for o in ["Vocals", "Bass", "Other", "Drums", "All"]):
            stems, _ = demucs_separate(wav_path, out_dir)

        # Refine drums into Kick, Snare, Hi-Hat (always run when drums are available)
        if "drums" in stems and os.path.exists(stems["drums"]):
            try:
                drum_refined = refine_drums(stems["drums"], out_dir)
            except Exception as d_err:
                print(f"[WARN] Drum refinement error: {d_err}")
                drum_refined = {}

        # Decoupled, guaranteed Key and Tempo detection
        key = "C Major"
        tempo = "120"
        try:
            import librosa
            # Sample first 35 seconds of the song for acoustic analysis
            y_full, sr_load = librosa.load(wav_path, mono=True, sr=22050, duration=35.0)

            # 1. Independent Tempo detection (test drums energy first, fallback to song mix)
            try:
                detected_bpm = None
                if "drums" in stems and os.path.exists(stems["drums"]):
                    try:
                        y_d, sr_d = librosa.load(stems["drums"], mono=True, sr=22050, duration=35.0)
                        if np.max(np.abs(y_d)) > 0.04:
                            detected_bpm = detect_tempo_advanced(y_d, sr_d)
                    except Exception:
                        pass
                if detected_bpm is None or detected_bpm <= 0:
                    detected_bpm = detect_tempo_advanced(y_full, sr_load)
                tempo = str(detected_bpm)
            except Exception as t_err:
                print(f"[WARN] Tempo detection error: {t_err}")
                tempo = "120"

            # 2. Independent Key detection (Essentia KeyExtractor with Krumhansl fallback)
            try:
                key = detect_key_advanced(y_full, sr_load)
            except Exception as k_err:
                print(f"[WARN] Key detection error: {k_err}")
                key = "C Major"

        except Exception as audio_load_err:
            print(f"[WARN] Audio load for key/tempo error: {audio_load_err}")

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

# --- Generative Instrument Endpoint ---
def process_generate_instrument(prompt, tempo="120", key="C Major", section="chorus", duration=15.0, ref_audio=None):
    if not prompt or not str(prompt).strip():
        return None, "ERROR: Please provide an instrument description."
    try:
        ref_path = None
        if ref_audio:
            ref_path, _ = extract_file_info(ref_audio)
            
        out_wav, meta = generate_instrument_stem(
            prompt=str(prompt).strip(),
            tempo=str(tempo).strip() if tempo else None,
            key=str(key).strip() if key else None,
            section=str(section).strip() if section else None,
            duration_sec=float(duration) if duration else 15.0,
            ref_audio_path=ref_path
        )
        if not out_wav or not meta.get("success"):
            err_msg = meta.get("error", "Unknown error during instrument generation.")
            return None, f"ERROR: {err_msg}"
            
        report_json = json.dumps(meta)
        report_str = f"SUCCESS:{report_json}"
        return out_wav, report_str
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

    # Generative Instruments
    with gr.Tab("Generative"):
        gen_prompt = gr.Textbox(
            label="Instrument Description",
            placeholder="e.g. 808 sub bass with distortion, soulful rhodes chords, punchy trap brass"
        )
        with gr.Row():
            gen_tempo = gr.Textbox(value="120", label="Tempo (BPM)")
            gen_key = gr.Textbox(value="C Major", label="Musical Key")
            gen_section = gr.Dropdown(
                choices=["chorus", "verse", "intro", "outro", "full"],
                value="chorus",
                label="Section"
            )
            gen_duration = gr.Slider(5, 30, value=15, step=1, label="Duration (seconds)")
        gen_ref = gr.Audio(type="filepath", label="Reference Audio for Melodic Conditioning (Optional)")
        gen_btn = gr.Button("Generate Instrument Stem")
        gen_out_wav = gr.Audio(label="Generated Instrument WAV")
        gen_status = gr.Textbox(label="Status / Diagnostics")
        gen_btn.click(
            fn=process_generate_instrument,
            inputs=[gen_prompt, gen_tempo, gen_key, gen_section, gen_duration, gen_ref],
            outputs=[gen_out_wav, gen_status],
            api_name="generate_instrument"
        )

# Direct HTTP POST /generate-instrument route on underlying FastAPI server
try:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    if hasattr(app, "app") and app.app is not None:
        @app.app.post("/generate-instrument")
        async def http_generate_instrument(request: Request):
            try:
                body = await request.json()
            except Exception:
                body = {}
            prompt = body.get("prompt") or body.get("description", "")
            tempo = body.get("tempo", "120")
            key = body.get("key", "C Major")
            section = body.get("section", "chorus")
            duration = float(body.get("duration", 15.0))
            ref_audio = body.get("ref_audio")
            
            out_wav, meta = generate_instrument_stem(
                prompt=prompt,
                tempo=tempo,
                key=key,
                section=section,
                duration_sec=duration,
                ref_audio_path=ref_audio
            )
            if not out_wav or not meta.get("success"):
                return JSONResponse(status_code=400, content={"status": "error", "error": meta.get("error")})
            return JSONResponse(status_code=200, content={"status": "success", "output_wav": out_wav, "metadata": meta})
except Exception:
    pass

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
