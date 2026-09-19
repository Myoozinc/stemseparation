# Myooz Tool Box — Audio Intelligence Suite

A professional studio production suite for modern creators:
- **Stemer**: Multi-stem separation powered by Demucs with key & BPM detection.
- **Midifier**: Polyphonic neural audio-to-MIDI transcription.
- **Sampler**: Transient/beat slicing and 8-pad MPC drum kit generator.
- **Mixter**: Adaptive multi-stem mixing with dynamic resonance suppression and musical section automation.
- **Master**: 5-stage studio mastering with closed-loop ITU-R BS.1770-4 LUFS measurement and correction (target $\pm 0.3$ LUFS).
- **Generative**: Neural instrument and arrangement generation via dedicated Hugging Face Inference Endpoints.

---

## Generative Engine & Hugging Face Inference Endpoints

The generative instrument module (`backend/generative_engine.py`) connects to a **dedicated Hugging Face Inference Endpoint** running music generation models such as `facebook/musicgen-melody` or `stabilityai/stable-audio-open-1.0`.

> [!NOTE]
> The generative module is completely modular and optional. If no endpoint credentials are configured, the engine operates in local mock synthesis mode without breaking Mixter or Master pipelines.

### Setup Instructions

1. **Create an Inference Endpoint:**
   - Navigate to [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints).
   - Click **New Endpoint**.
   - **Model Repository**: Enter `facebook/musicgen-melody` (or `stabilityai/stable-audio-open-1.0`).
   - **Task**: Audio Generation / Text-to-Audio.
   - **Hardware**: Select a GPU instance (e.g., Nvidia T4 Small or A10G).
   - **Automatic Scaling**: Enable **Scale to 0** (minimum 0 replicas). This ensures the instance automatically shuts down when idle, minimizing cost.
   - Click **Create Endpoint**.

2. **Configure Environment Variables:**
   Obtain your endpoint URL and a Hugging Face User Access Token (with Read permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

   ```bash
   export HF_ENDPOINT_URL="https://your-endpoint-name.endpoints.huggingface.cloud"
   export HF_API_TOKEN="hf_yourAccessTokenHere"
   ```

   Optional configuration:
   ```bash
   export HF_TIMEOUT="75"   # Timeout in seconds (allows for GPU cold-start)
   ```

3. **API Usage:**
   - **HTTP POST**: `/generate-instrument`
     ```json
     {
       "prompt": "808 sub bass with saturation",
       "tempo": "130",
       "key": "F# Minor",
       "section": "chorus",
       "duration": 15
     }
     ```
   - **Gradio Client**: Exposes `generate_instrument` on port 7860.
   - Outputs a **32-bit Float WAV** stem at 44.1 kHz normalized to -3.0 dBFS true headroom, ready to drag directly into Mixter.

