"""
Live Audio Transcription using Google Cloud Speech-to-Text V2 (Chirp 3)

This script captures audio from your microphone and sends it to the Chirp 3 model
via the Google Cloud Speech-to-Text V2 API. It uses Application Default Credentials (ADC)
for authentication.

Features configured:
- Chirp 3 Model
- Portuguese (Brazil) language
- Voice Activity Timeout (to address endpointing latency)
- Interim Results (to see partial transcriptions)

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
import pyaudio
import queue
from dotenv import load_dotenv

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from google.protobuf.duration_pb2 import Duration

# Load environment variables
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = "us"  # Chirp 3 is available in 'us' region (global endpoint usually redirects or explicit)

if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT not found in .env file or environment.")

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk, device_index=None):
        self._rate = rate
        self._chunk = chunk
        self._device_index = device_index
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
            input_device_index=self._device_index,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            
            # Optional debug: print dot for every chunk to verify audio flow
            print(".", end="", flush=True)
            yield b"".join(data)

def list_microphones(pya):
    """Lists available microphone devices."""
    info = pya.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    print("\nAvailable Audio Input Devices:")
    input_devices = []
    for i in range(0, numdevices):
        if (pya.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = pya.get_device_info_by_host_api_device_index(0, i).get('name')
            print(f"ID {i}: {name}")
            input_devices.append(i)
    return input_devices

def transcribe_streaming_chirp3_mic():
    """Transcribes audio from microphone using Chirp 3."""
    
    # Microphone selection
    pya = pyaudio.PyAudio()
    try:
        available_ids = list_microphones(pya)
        if not available_ids:
            print("No microphone devices found.")
            return

        try:
            selection = input("\nEnter microphone ID to use (default is system default): ")
            if selection.strip():
                device_index = int(selection)
                if device_index not in available_ids:
                    print(f"Invalid ID {device_index}. Using default.")
                    device_index = None
            else:
                device_index = None
        except ValueError:
            print("Invalid input. Using default microphone.")
            device_index = None
    finally:
        pya.terminate()

    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{REGION}-speech.googleapis.com",
        )
    )

    # Configuration
    recognition_config = cloud_speech.RecognitionConfig(
        # IMPORTANT: We must use ExplicitDecodingConfig for raw microphone audio.
        # AutoDetectDecodingConfig fails because raw PCM (LINEAR16) has no header 
        # (unlike WAV), so the API cannot auto-detect format and hangs.
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            audio_channel_count=CHANNELS,
        ),
        language_codes=["pt-BR"], # Portuguese (Brazil)
        model="chirp_3",
    )
    
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            # interim_results: Returns partial transcripts (is_final=False) as you speak.
            # Critical for user feedback so they know the system is listening.
            interim_results=True, 
            
            # enable_voice_activity_events: Returns events when speech starts/ends.
            enable_voice_activity_events=True,
            
            # voice_activity_timeout: SOLVES CHIRP 3 LATENCY ISSUE.
            # Chirp 3 can be slow to finalize short utterances (like "Tudo").
            # Setting speech_end_timeout forces the model to finalize the result 
            # if it detects silence for X seconds (here 2s), preventing 10s+ delays.
            # We use a dictionary because the VoiceActivityTimeout class might not 
            # be directly exposed in all library versions.
            voice_activity_timeout={
                "speech_end_timeout": {"seconds": 2}
            }
        ),
    )

    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{REGION}/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests(config: cloud_speech.StreamingRecognizeRequest, audio: list) -> list:
        yield config
        yield from audio

    print(f"\n🎙️  Listening (Chirp 3 - pt-BR)... Press Ctrl+C to stop.")
    print(f"Project: {PROJECT_ID}")
    if device_index is not None:
        print(f"Using Microphone ID: {device_index}")
    else:
        print("Using Default Microphone")
    print("=======================================================\n")

    with MicrophoneStream(SAMPLE_RATE, CHUNK_SIZE, device_index=device_index) as stream:
        audio_generator = stream.generator()
        
        # Generator yielding StreamingRecognizeRequest objects for audio chunks
        audio_requests = (
            cloud_speech.StreamingRecognizeRequest(audio=content) 
            for content in audio_generator
        )

        try:
            # Transcribes the audio into text
            responses = client.streaming_recognize(
                requests=requests(config_request, audio_requests)
            )
            print("Connection established. Waiting for results...")

            # Process responses
            for response in responses:
                # print(f"Debug: Response received: {response}") # Uncomment for deep debugging
                if response.speech_event_type == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                    print("\r📢 Speech started", end="", flush=True)
                if response.speech_event_type == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                    print("\r📢 Speech ended", end="", flush=True)

                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                
                # Check if it's a final result or interim
                if result.is_final:
                    print(f"\r✅ Final: {transcript}")
                else:
                    # Print interim results in place
                    print(f"\r⏳ Interim: {transcript}", end="", flush=True)

        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    try:
        transcribe_streaming_chirp3_mic()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user.")
