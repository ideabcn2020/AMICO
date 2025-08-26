# amico_id_policy.py
from __future__ import annotations
from typing import Tuple, List
from amico_id_types import Evidence, Decision, Action, PolicyConfig, SessionState

def classify_voice(score: float, cfg: PolicyConfig) -> Tuple[bool,bool]:
    return (score >= cfg.voice_strong, score >= cfg.voice_ok)

def classify_face(score: float, cfg: PolicyConfig) -> Tuple[bool,bool]:
    return (score >= cfg.face_strong, score >= cfg.face_ok)

def decide_identity(voice: Evidence, face: Evidence, cfg: PolicyConfig) -> Decision:
    # Agreement path
    if voice.user_id and face.user_id and voice.user_id == face.user_id:
        if (voice.strong and face.ok) or (face.strong and voice.ok):
            return Decision(voice.user_id, "both", max(voice.score, face.score), "strong")
    # Single strong path
    if voice.strong and not face.strong:
        return Decision(voice.user_id, "voice", voice.score, "strong" if voice.ok else "weak")
    if face.strong and not voice.strong:
        return Decision(face.user_id, "face", face.score, "strong" if face.ok else "weak")
    # One OK → pick higher (weak)
    if voice.ok or face.ok:
        best = voice if voice.score >= face.score else face
        return Decision(best.user_id, best.src, best.score, "weak")
    # Unknown
    return Decision(None, "none", 0.0, "unknown")

def plan_actions(decision: Decision, voice: Evidence, face: Evidence,
                 state: SessionState, cfg: PolicyConfig) -> List[Action]:
    acts: List[Action] = []

    # Continual learning: if we recognized someone, store fresh good-quality samples
    if decision.user_id:
        uid = decision.user_id
        if voice.user_id == uid and voice.score >= cfg.store_voice_min_sim:
            acts.append(Action("STORE_VOICE", {"user_id": uid, "score": voice.score}))
        if face.user_id == uid and face.score >= cfg.store_face_min_sim:
            meta = {k: face.meta.get(k) for k in ("det_score","bbox","faces_count") if k in face.meta}
            acts.append(Action("STORE_FACE", {"user_id": uid, "score": face.score, **meta}))
        return acts

    # Unknown → politely ask for name (rate-limited)
    if state.can_ask_name(cfg):
        acts.append(Action("ASK_NAME", {
            "prompt": "Hola 👋, no te tengo fichado aún. ¿Cómo te llamas?",
            "lang": "es"
        }))
        state.mark_asked()
    else:
        acts.append(Action("LOG_ONLY", {"msg": "Unknown this turn; cooldown active"}))
    return acts
