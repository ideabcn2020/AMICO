# amico.py — clean console + split warning capture
from pathlib import Path
import warnings
from contextlib import redirect_stderr

# ---------- LOG SETUP (run this BEFORE any noisy imports) ----------
LOG_DIR = (Path(__file__).parent / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMPORT_WARN_LOG = LOG_DIR / "import_warnings.log"   # warnings + stderr during imports
WARN_LOG        = LOG_DIR / "warnings.log"          # warnings during runtime

# Truncate both logs on each run
IMPORT_WARN_LOG.write_text("")
WARN_LOG.write_text("")

def _make_showwarning(target_path: Path):
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(warnings.formatwarning(message, category, filename, lineno, line))
    return _showwarning

# Ensure warnings are emitted so we can capture them
warnings.simplefilter("default")

# During imports: capture warnings to IMPORT_WARN_LOG and redirect stderr there too
warnings.showwarning = _make_showwarning(IMPORT_WARN_LOG)
with open(IMPORT_WARN_LOG, "a", encoding="utf-8") as _imp_err, redirect_stderr(_imp_err):
    from amico_stt import stt
    from amico_listen import record_audio
    from amico_vp import vp, _valid_vp
    from amico_emotions import detect_emotion
    from amico_fuse_emotion import fuse_audio_text

# After imports: route all subsequent warnings to WARN_LOG
warnings.showwarning = _make_showwarning(WARN_LOG)
# ---------- END LOG SETUP ----------


def main():
    print("🤖 AMICO is running. Press Ctrl+C to stop.")
    try:
        while True:
            print("🎙️ AMICO v0.2")

            audio_path = record_audio()
            input("Press Enter to continue...")

            print("📝 Transcribing...")
            text, language = stt(audio_path)
            print(f"🗣️ You said: {text}")
            print(f"🌐 Detected language: {language}")
            input("Press Enter to continue...")
            
            print("📝 Emotion")
            emo_a = detect_emotion(audio_path, mode="light")  # audio arousal (cheap)  :contentReference[oaicite:4]{index=4}
            txt = (text or "").strip()
            if len(txt) >= 8:
                try:
                    # lazy import here, AFTER your import-redirect block is long gone
                    from amico_txt_emotion import detect_text_emotion
                    # capture progress bars / logs into WARN_LOG
                    with open(WARN_LOG, "a", encoding="utf-8") as _runlog, redirect_stderr(_runlog), redirect_stdout(_runlog):
                        te = detect_text_emotion(txt, lang=language or "en")
                    a_fused, lab_fused = fuse_audio_text(emo_a["arousal"], te["dist"])
                    print(f"🙂 audio:{emo_a['label']}({emo_a['arousal']:.2f})  📝 text:{te['label']}({te['confidence']:.2f})  🧪 fused:{lab_fused}({a_fused:.2f})")
                except Exception:
                    # transformers missing / first-run download error → stay audio-only
                    print(f"🙂 audio-only:{emo_a['label']}({emo_a['arousal']:.2f}) — text model unavailable")
            else:
                print(f"🙂 audio-only:{emo_a['label']}({emo_a['arousal']:.2f}) — text too short")

            print("🗣️ Temp store of 'You said' for storage purpose")
            user_said = text  # placeholder for storage module
            input("Press Enter to continue...")

            print("📝 Extracting Voiceprint...")
            voiceprint = vp(audio_path)
            if _valid_vp(voiceprint):
                print("✅ Voiceprint extracted")
            else:
                print("⚠️ No voiceprint extracted")

            input("Press Enter to continue...")
            quit()

    except KeyboardInterrupt:
        print("\n👋 Shutting down AMICO.")


if __name__ == "__main__":
    main()
