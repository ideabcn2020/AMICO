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
