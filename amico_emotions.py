# amico_emotions.py
import numpy as np
import torch
import torchaudio
import soundfile as sf

# ---------- utils ----------
def _load_mono(path: str) -> tuple[torch.Tensor, int]:
    # (frames, ch) float32
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # (ch, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr

def _to_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    # peak normalize
    m = wav.abs().max()
    if m > 0:
        wav = wav / m
    return wav

# ---------- LIGHT mode (prosody) ----------
def _feature_rms(wav: torch.Tensor) -> float:
    # RMS energy
    return float(torch.sqrt((wav**2).mean()).item())

def _feature_centroid(wav: torch.Tensor, sr: int) -> float:
    # simple spectral centroid over the whole clip
    x = wav.squeeze(0)
    n = 1 << (x.numel()-1).bit_length()  # next pow2
    x = torch.nn.functional.pad(x, (0, max(0, n - x.numel())))
    X = torch.fft.rfft(x)
    mag = X.abs() + 1e-9
    freqs = torch.linspace(0, sr/2, steps=mag.numel())
    centroid = float((freqs * mag).sum() / mag.sum())
    return centroid

def _arousal_from_prosody(wav: torch.Tensor, sr: int) -> float:
    # normalize RMS & centroid to z-scores using rough speech priors
    rms = _feature_rms(wav)
    cen = _feature_centroid(wav, sr)
    # priors (rough): rms ~ 0.05 ± 0.03 ; centroid ~ 2000 ± 800 Hz
    z_rms = (rms - 0.05) / 0.03
    z_cen = (cen - 2000.0) / 800.0
    z = 0.6 * z_rms + 0.4 * z_cen
    # squash to 0..1
    arousal = float(1 / (1 + np.exp(-z)))
    return max(0.0, min(1.0, arousal))

def _label_from_arousal(a: float) -> str:
    if a >= 0.66:
        return "excited"
    if a <= 0.33:
        return "calm"
    return "neutral"

# ---------- HF mode (transformers) ----------
def _predict_hf(wav16: torch.Tensor):
    """
    Returns: dict(label, scores={label: prob, ...})
    Model: superb/hubert-base-superb-er
    """
    # lazy import inside function to keep startup clean
    from transformers import AutoProcessor, AutoModelForAudioClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "superb/hubert-base-superb-er"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id).to(device)
    with torch.no_grad():
        inputs = processor(wav16.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    id2label = model.config.id2label
    scores = {id2label[i]: float(p) for i, p in enumerate(probs)}
    label = max(scores, key=scores.get)
    return {"label": label, "scores": scores}

# ---------- Public API ----------
def detect_emotion(audio_path: str, mode: str = "light") -> dict:
    """
    Returns a dict with:
      - mode: "light" or "hf"
      - label: str
      - arousal: float in [0,1]  (always present; for HF it's mapped from probs)
      - scores: dict(label->prob) (HF only)
    """
    wav, sr = _load_mono(audio_path)
    wav16 = _to_16k(wav, sr)

    if mode == "hf":
        try:
            out = _predict_hf(wav16)
            # map to an arousal proxy from probabilities (excited > calm)
            emo = out["label"]
            scores = out["scores"]
            arousal = (
                scores.get("angry", 0.0) * 0.9 +
                scores.get("happy", 0.7) +
                scores.get("neutral", 0.4) +
                scores.get("sad", 0.1)
            )
            arousal = float(max(0.0, min(1.0, arousal)))
            return {"mode": "hf", "label": emo, "arousal": arousal, "scores": scores}
        except Exception:
            # fall back to light on any issue (e.g., transformers not installed)
            pass

    # LIGHT fallback / default
    a = _arousal_from_prosody(wav16, 16000)
    return {"mode": "light", "label": _label_from_arousal(a), "arousal": a}
