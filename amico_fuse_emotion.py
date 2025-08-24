# fuse_emotion.py
def fuse_audio_text(arousal: float, tdist: dict[str, float]):
    """
    tdist: probabilities from text model. Keys are lowercase like:
      anger, disgust, fear, joy, neutral, sadness, surprise, ...
    """
    # positives raise arousal
    pos = 0.9 * tdist.get("anger", 0.0) \
        + 0.7 * tdist.get("surprise", 0.0) \
        + 0.6 * tdist.get("joy", 0.0)
    # negatives lower arousal
    neg = 0.5 * tdist.get("sadness", 0.0) \
        + 0.4 * tdist.get("fear", 0.0) \
        + 0.3 * tdist.get("disgust", 0.0)
    t_arousal = 0.5 + pos - neg                     # center at 0.5
    t_arousal = max(0.0, min(1.0, t_arousal))       # clamp

    a = 0.7 * float(arousal) + 0.3 * float(t_arousal)  # 70% audio, 30% text
    if a >= 0.70:
        lab = "excited"
    elif a <= 0.30:
        lab = "calm"
    else:
        lab = "neutral"
    return a, lab
