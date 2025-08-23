import numpy as np
import torch, torchaudio
from speechbrain.pretrained import EncoderClassifier

# Modelo de extracción
MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_enc = None

def _model():
    global _enc
    if _enc is None:
        _enc = EncoderClassifier.from_hparams(source=MODEL, run_opts={"device": DEVICE})
    return _enc
    
def _mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)                 # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    m = wav.abs().max()
    if m > 0:
        wav = wav / m
    return wav

@torch.no_grad()
def vp(path: str) -> np.ndarray:
    wav = _mono_16k(path).to(DEVICE)
    emb = _model().encode_batch(wav)                # (1, 192)
    emb = emb.squeeze(0).squeeze(0).detach().cpu().numpy().astype("float32")
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n                               # L2-normalize
    return emb                                      # shape (192,)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

