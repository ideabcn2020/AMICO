# amico_identity.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import numpy as np
from amico_id_types import PolicyConfig, SessionState, Evidence, Decision
from amico_id_policy import classify_voice, classify_face, decide_identity, plan_actions

# Provider signatures (you’ll inject your own functions)
VoiceExtract = Callable[[str], np.ndarray]                      # audio_path -> (192,) float32
VoiceMatch   = Callable[[np.ndarray], tuple[Optional[str], float]]

FaceCapture  = Callable[[], Any]                                # -> frame (BGR ndarray)
FaceExtract  = Callable[[Any], List[Dict[str,Any]]]             # frame -> list[{emb:(512,), score, bbox}]
FaceMatch    = Callable[[np.ndarray], tuple[Optional[str], float]]

class IdentityOrchestrator:
    def __init__(self,
                 voice_extract: VoiceExtract,
                 voice_match: VoiceMatch,
                 face_capture: FaceCapture,
                 face_extract: FaceExtract,
                 face_match: FaceMatch,
                 cfg: PolicyConfig,
                 state: SessionState):
        self.voice_extract = voice_extract
        self.voice_match   = voice_match
        self.face_capture  = face_capture
        self.face_extract  = face_extract
        self.face_match    = face_match
        self.cfg = cfg
        self.state = state

    def identify_turn(self, audio_path: str, frame_bgr=None) -> Decision:
        # 1) Voice
        v_emb = self.voice_extract(audio_path)            # (192,)
        v_uid, v_score = self.voice_match(v_emb)
        v_strong, v_ok = classify_voice(v_score, self.cfg)
        v_ev = Evidence("voice", v_uid, v_score, v_strong, v_ok)

        # 2) Faces: match ALL faces, take the best match (if any)
        if frame_bgr is None:
            frame_bgr = self.face_capture()
        faces = self.face_extract(frame_bgr) or []
        best_uid, best_score, best_meta = None, 0.0, {}
        for f in faces:
            uid, s = self.face_match(f["emb"])
            if s > best_score:
                best_uid, best_score = uid, s
                best_meta = {"det_score": f.get("score"), "bbox": f.get("bbox"), "faces_count": len(faces)}
        f_strong, f_ok = classify_face(best_score, self.cfg)
        f_ev = Evidence("face", best_uid, best_score, f_strong, f_ok, meta=best_meta)

        # 3) Decide + plan actions
        decision = decide_identity(v_ev, f_ev, self.cfg)
        decision.actions = plan_actions(decision, v_ev, f_ev, self.state, self.cfg)
        if decision.user_id:
            self.state.last_user_id = decision.user_id
        return decision
