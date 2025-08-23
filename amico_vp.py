import torchaudio

# Modelo de extracción
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec"
)

# Extraemos voiceprint
def vp(audio_path):
    signal, sr = torchaudio.load(audio_path)

    # If stereo, convert to mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    embedding = spkrec.encode_batch(signal)
    return embedding.squeeze()
