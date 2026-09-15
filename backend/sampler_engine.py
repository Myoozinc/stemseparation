"""
Sampler Engine: Audio Slicer & Sample Kit Generator
Extracts one-shots, beats, or vocal chops with transient detection,
and exports organized sample packs in a downloadable ZIP archive.
"""
import os
import zipfile
import numpy as np
import soundfile as sf

def slice_audio_samples(audio_path, output_dir=None, mode="transients", max_samples=16, min_duration=0.15):
    """
    Slices audio into musical samples.
    Args:
        audio_path (str): Path to audio file
        output_dir (str, optional): Destination folder
        mode (str): 'transients' for one-shots or 'beats' for rhythmic slices
        max_samples (int): Max number of samples to generate
        min_duration (float): Minimum length in seconds per sample
    Returns:
        tuple: (zip_path, list_of_sample_info)
    """
    import librosa

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(audio_path) or ".", "samples_output")
    os.makedirs(output_dir, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=None, mono=False)
    # Check mono vs stereo
    is_stereo = len(y.shape) > 1 and y.shape[0] == 2
    y_mono = librosa.to_mono(y) if is_stereo else y

    # Detect onsets or beats
    if mode == "beats":
        tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
        slice_points = librosa.frames_to_samples(beat_frames)
    else:
        # Transient / Onset slicing
        onset_frames = librosa.onset.onset_detect(
            y=y_mono,
            sr=sr,
            backtrack=True,
            pre_max=20,
            post_max=20,
            pre_avg=100,
            post_avg=100,
            delta=0.07,
            wait=int(sr * min_duration / 512)
        )
        slice_points = librosa.frames_to_samples(onset_frames)

    # Ensure start at 0
    if len(slice_points) == 0 or slice_points[0] != 0:
        slice_points = np.insert(slice_points, 0, 0)
    
    total_samples = y.shape[-1]
    slice_points = np.append(slice_points, total_samples)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    generated_files = []
    sample_info_list = []

    min_sample_len = int(min_duration * sr)

    sample_count = 0
    for i in range(len(slice_points) - 1):
        start = slice_points[i]
        end = slice_points[i + 1]

        if (end - start) < min_sample_len:
            continue

        if is_stereo:
            chunk = y[:, start:end]
            # Fade out last 50ms to prevent click
            fade_len = min(int(0.05 * sr), chunk.shape[1] // 4)
            if fade_len > 0:
                fade = np.linspace(1, 0, fade_len)
                chunk[:, -fade_len:] *= fade
        else:
            chunk = y[start:end]
            fade_len = min(int(0.05 * sr), len(chunk) // 4)
            if fade_len > 0:
                fade = np.linspace(1, 0, fade_len)
                chunk[-fade_len:] *= fade

        # Peak normalize chunk
        peak = np.max(np.abs(chunk))
        if peak > 0.001:
            chunk = chunk / peak * 0.95

        sample_count += 1
        sample_name = f"{base_name}_sample_{sample_count:02d}.wav"
        sample_file_path = os.path.join(output_dir, sample_name)

        if is_stereo:
            sf.write(sample_file_path, chunk.T, sr)
        else:
            sf.write(sample_file_path, chunk, sr)

        dur = (end - start) / sr
        generated_files.append(sample_file_path)
        sample_info_list.append({
            "name": sample_name,
            "path": sample_file_path,
            "duration": round(dur, 2)
        })

        if sample_count >= max_samples:
            break

    # Package into a ZIP
    zip_path = os.path.join(output_dir, f"{base_name}_sample_pack.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fpath in generated_files:
            zipf.write(fpath, os.path.basename(fpath))

    return zip_path, sample_info_list
