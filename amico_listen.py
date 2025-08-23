import tempfile
import sounddevice #as sd
import soundfile #as sf


# AVariable que indica el nombre del micrófono que queremos utilizar
PREFERRED_MIC_NAME = "Razer Seiren V3 Mini"

# Esta función nos da el index del micrófono qeu queremos
def get_preferred_mic_index(preferred_name=PREFERRED_MIC_NAME):
    """Return index of preferred mic or fallback to default input"""
    for idx, device in enumerate(sounddevice.query_devices()):
        if preferred_name.lower() in device['name'].lower() and device['max_input_channels'] > 0:
            return idx
    return sounddevice.default.device[0]  # fallback to default input

# Esta función graba el audio en un archivo temporal en foramto wav. Este archivo es para aplicar el STT
def record_audio(duration=5, samplerate=16000):
    """Record audio from preferred mic and save to a temp WAV file"""
    mic_index = get_preferred_mic_index()
    print(f"🎤 Using mic: {sounddevice.query_devices()[mic_index]['name']}")

    print("🎙️ Listening... Speak now.")
    recording = sounddevice.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=mic_index)
    sounddevice.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    soundfile.write(temp_file.name, recording, samplerate)
    print(f"🔊 Audio saved at: {temp_file.name}")
    return temp_file.name
