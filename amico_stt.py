import whisper

#Load whisper model
model = whisper.load_model("base")

# Transcribe el audio en el archivo audio_path y reconoce el idioma.
def stt(audio_path):
    result = model.transcribe(audio_path)
    text = result.get("text", "").strip()
    language = result.get("language", "und")  # 'und' means undefined
    return text, language
