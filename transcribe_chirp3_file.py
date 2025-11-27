"""
File Audio Transcription using Google Cloud Speech-to-Text V2 (Chirp 3)

This script reads audio from 'teste2.wav' and sends it to the Chirp 3 model
via the Google Cloud Speech-to-Text V2 API.

Features configured:
- Chirp 3 Model
- Portuguese (Brazil) language
- Voice Activity Timeout
- Interim Results

## Setup

1. Ensure you have Google Cloud SDK installed and authenticated:
   ```
   gcloud auth application-default login
   ```
2. Ensure requirements are installed:
   ```
   pip install -r requirements.txt
   ```
3. Set GOOGLE_CLOUD_PROJECT in .env (or environment).
"""

import os
import wave
import time
import threading
from dotenv import load_dotenv

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

# Load environment variables
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = "us"

if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT not found in .env file or environment.")

# Audio configuration
AUDIO_FILE = "teste2.wav"
CHUNK_SIZE = 1024

# Shared state for tracking audio sent
class AudioTracker:
    def __init__(self):
        self.audio_sent_seconds = 0.0
        self.start_time = None

    def get_elapsed_time(self):
        if self.start_time is None:
            return "0.00s"
        return f"{time.time() - self.start_time:.2f}s"

tracker = AudioTracker()

def stream_file(file_path, chunk_size):
    """Generator that yields audio chunks from a WAV file."""
    with wave.open(file_path, "rb") as wf:
        sample_rate = wf.getframerate()
        # Calculate how much time one chunk represents in seconds
        chunk_duration = chunk_size / sample_rate
        
        data = wf.readframes(chunk_size)
        while len(data) > 0:
            yield data
            # Update tracker
            tracker.audio_sent_seconds += chunk_duration
            
            # Simulate real-time streaming to match the audio playback speed
            time.sleep(chunk_duration)
            data = wf.readframes(chunk_size)

def transcribe_streaming_chirp3_file():
    """Transcribes audio from file using Chirp 3."""
    
    # Check if file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found.")
        return

    # Get file properties to configure the recognizer correctly
    with wave.open(AUDIO_FILE, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        print(f"Audio File: {AUDIO_FILE}")
        print(f"Sample Rate: {sample_rate}, Channels: {channels}")

    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{REGION}-speech.googleapis.com",
        )
    )

    # Configuration
    recognition_config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            audio_channel_count=channels,
        ),
        language_codes=["pt-BR"],
        model="chirp_3",
    )
    
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True,
            enable_voice_activity_events=True,
            voice_activity_timeout={
                "speech_end_timeout": {"seconds": 2}
            }
        ),
    )

    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{REGION}/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests(config: cloud_speech.StreamingRecognizeRequest, audio_generator) -> list:
        yield config
        for content in audio_generator:
            yield cloud_speech.StreamingRecognizeRequest(audio=content)

    print(f"\n🚀 Starting transcription for {AUDIO_FILE} (Chirp 3 - pt-BR)...")
    print(f"Project: {PROJECT_ID}")
    print("=======================================================\n")

    try:
        # Start tracking time
        tracker.start_time = time.time()
        
        # Generator yielding audio chunks
        audio_generator = stream_file(AUDIO_FILE, CHUNK_SIZE)

        # Transcribes the audio into text
        responses = client.streaming_recognize(
            requests=requests(config_request, audio_generator)
        )
        print("Connection established. Processing...")

        for response in responses:
            elapsed = tracker.get_elapsed_time()
            sent = f"{tracker.audio_sent_seconds:.2f}s"
            
            # Handle Speech Events
            if response.speech_event_type == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                print(f"\n[SpeechActivityBegin] {elapsed} | Audio sent: {sent}")
                continue # No transcript in this event usually
                
            if response.speech_event_type == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                print(f"\n[SpeechActivityEnd] {elapsed} | Audio sent: {sent}")
                continue

            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            stability = result.stability
            
            if result.is_final:
                print(f"[FinalResults]      {elapsed} | Transcription: {transcript} | isFinal: True | Audio sent: {sent}")
            else:
                print(f"[InterimResults]    {elapsed} | Transcription: {transcript} | Stability: {stability:.2f} | Audio sent: {sent}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    try:
        transcribe_streaming_chirp3_file()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user.")
