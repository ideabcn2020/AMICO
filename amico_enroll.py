# amico_enroll.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List
import numpy as np

# Simple name extractor fallback (replace with your LLM later)
def extract_name_simple(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text: return None
    # naive: take first 2 words
    parts = [p for p in text.replace(",", " ").split() if p.isalpha()]
    if not parts: return None
    name = " ".join(parts[:2])
    return name.title()

@dataclass
class EnrollState:
    collecting: bool = False
    user_id: Optional[str] = None
    voice_embs: List[np.ndarray] = field(default_factory=list)
    face_embs:  List[np.ndarray] = field(default_factory=list)
    needed_voice: int = 3
    needed_face: int = 3

class EnrollmentManager:
    def __init__(self): self.state = EnrollState()

    def start(self, name_text: str) -> Optional[str]:
        uid = extract_name_simple(name_text)
        if not uid: return None
        self.state = EnrollState(collecting=True, user_id=uid)
        return uid

    def add_voice(self, emb: np.ndarray):
        if self.state.collecting: self.state.voice_embs.append(emb)

    def add_face(self, emb: np.ndarray):
        if self.state.collecting: self.state.face_embs.append(emb)

    def done(self) -> bool:
        s = self.state
        return s.collecting and len(s.voice_embs) >= s.needed_voice and len(s.face_embs) >= s.needed_face

    def finish(self):
        s = self.state
        uid = s.user_id
        v = list(s.voice_embs); f = list(s.face_embs)
        self.state = EnrollState()
        return uid, v, f  # you will STORE_* these via your DB adapters
