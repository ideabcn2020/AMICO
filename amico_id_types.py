# amico_id_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict
import time

# Thresholds & behavior (tune later)
@dataclass
class PolicyConfig:
    voice_ok: float = 0.65
    voice_strong: float = 0.80
    face_ok: float = 0.45
    face_strong: float = 0.60
    store_voice_min_sim: float = 0.78   # only store new sample if >= this
    store_face_min_sim: float = 0.55
    ask_name_cooldown_s: int = 90       # don’t nag
    max_voiceprints_per_user: int = 10
    max_faceprints_per_user: int = 10

# Evidence from one modality
@dataclass
class Evidence:
    src: Literal["voice","face"]
    user_id: Optional[str]          # best match user or None
    score: float                    # cosine similarity
    strong: bool
    ok: bool
    meta: Dict = field(default_factory=dict)  # e.g., det_score, bbox, faces_count

# What the logic wants the app to do (you’ll execute these)
@dataclass
class Action:
    kind: Literal["STORE_VOICE","STORE_FACE","ASK_NAME","BEGIN_ENROLL","LOG_ONLY"]
    data: Dict = field(default_factory=dict)

# Final decision for this turn
@dataclass
class Decision:
    user_id: Optional[str]                 # chosen identity (or None)
    via: Literal["both","voice","face","none"]
    confidence: float
    certainty: Literal["strong","weak","unknown"]
    actions: List[Action] = field(default_factory=list)

# Session memory across turns (for politeness & smoothing)
@dataclass
class SessionState:
    last_user_id: Optional[str] = None
    last_ask_ts: float = 0.0

    def can_ask_name(self, cfg: PolicyConfig) -> bool:
        return (time.time() - self.last_ask_ts) >= cfg.ask_name_cooldown_s

    def mark_asked(self):
        self.last_ask_ts = time.time()
