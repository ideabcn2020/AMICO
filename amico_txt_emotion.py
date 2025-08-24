# amico_txt_emotion.py
from functools import lru_cache
import os

def _quiet_hf():
    # kill bars & logs BEFORE creating pipelines
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

@lru_cache(maxsize=1)
def _pipe_en():
    _quiet_hf()
    from transformers import pipeline  # lazy import
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None, return_all_scores=True)

@lru_cache(maxsize=1)
def _pipe_multi():
    _quiet_hf()
    from transformers import pipeline  # lazy import
    return pipeline("text-classification",
                    model="joeddav/distilbert-base-multilingual-cased-go-emotions",
                    top_k=None, return_all_scores=True)

def _neutral():
    return {"label": "neutral", "confidence": 0.5, "dist": {"neutral": 1.0}, "model": "none"}

def detect_text_emotion(text: str, lang: str = "en"):
    text = (text or "").strip()
    if not text:
        return _neutral()
    try:
        use_en = (lang or "en").lower().startswith("en")
        pl = _pipe_en() if use_en else _pipe_multi()
        out = pl(text)[0]                            # list of {label, score}
        dist = {d["label"].lower(): float(d["score"]) for d in out}
        # Optional: renormalize multi-label outputs (GoEmotions)
        s = sum(dist.values()) or 1.0
        dist = {k: v / s for k, v in dist.items()}
        label = max(dist, key=dist.get)
        return {"label": label, "confidence": dist[label], "dist": dist, "model": "en" if use_en else "multi"}
    except Exception:
        return _neutral()
