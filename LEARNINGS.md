# Chirp 3 Streaming Transcription Learnings

This document summarizes key learnings and solutions found while implementing live streaming transcription with Google Cloud Speech-to-Text V2 (Chirp 3).

## 1. Endpointing Latency with Chirp 3
**Problem:** 
When using the Chirp 3 model (`model="chirp_3"`), the API can be "lazy" or conservative about finalizing transcription for short, quiet utterances (e.g., a simple "Tudo"). The model might wait up to 10 seconds of silence before deciding the speech has ended, causing significant latency in the final response.

**Solution:**
Use **Voice Activity Timeouts** to force the API to close the utterance after a specific duration of silence.

*   **Configuration:** Set `speech_end_timeout` within `streaming_features`.
*   **Effect:** If the API detects silence for the specified duration (e.g., 2 seconds) after speech, it forces the result to be finalized immediately.

```python
voice_activity_timeout={
    "speech_end_timeout": {"seconds": 2}
}
```

## 2. Raw Audio vs. Auto-Detection
**Problem:**
Using `AutoDetectDecodingConfig` with a live microphone stream causes the API to hang silently ("nothing is happening"). 
*   **Cause:** `AutoDetectDecodingConfig` expects audio data to include a header (like a WAV file header) to identify the format (Sample Rate, Encoding, Channels).
*   **Context:** Raw audio from `pyaudio` (Microphone) is headerless PCM bytes. The API receives bytes but waits indefinitely for a header that never arrives.

**Solution:**
Use **`ExplicitDecodingConfig`** to strictly define the audio format.

```python
explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    audio_channel_count=1,
)
```

## 3. Library Structure & Configuration
**Challenge:**
The `VoiceActivityTimeout` class might not be exposed at the top level of the `google.cloud.speech_v2` library in some versions, causing `AttributeError` if you try to instantiate it directly (e.g., `cloud_speech.StreamingRecognitionConfig.VoiceActivityTimeout`).

**Solution:**
Google Cloud Python client libraries support passing **dictionaries** for message fields.
*   Instead of importing the class, pass a dictionary structure matching the protobuf definition.
*   Ensure `voice_activity_timeout` is placed inside `streaming_features` (part of `StreamingRecognitionFeatures`), NOT directly in `StreamingRecognitionConfig`.

```python
streaming_features=cloud_speech.StreamingRecognitionFeatures(
    enable_voice_activity_events=True,
    voice_activity_timeout={
        "speech_end_timeout": {"seconds": 2}
    }
)
```

## 4. Interim Results
**Feature:**
Enabling `interim_results=True` allows the client to receive partial transcriptions ("...Tudo...") immediately as they are spoken, providing visual feedback to the user before the final result is settled. This is crucial for perceived responsiveness.
