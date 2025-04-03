"""
Live Audio Transcription using Gemini Live API

This script captures audio from your microphone and provides real-time
transcription using Google's Gemini Live API.

## Setup

To install the dependencies for this script, run:

```
pip install -r requirements.txt
```
"""

import asyncio
import pyaudio
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Gemini model configuration
MODEL = "models/gemini-2.0-flash-exp"

# Initialize PyAudio
pya = pyaudio.PyAudio()

# Configure Gemini client
client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=api_key)

# Configure for text-only responses (transcription)
CONFIG = types.LiveConnectConfig(
    response_modalities=["text"],
)


class TranscriptionLoop:
    def __init__(self):
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.transcription = ""
        self.is_running = True

    async def listen_audio(self):
        """Capture audio from microphone and send to queue"""
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        print("\n🎤 Listening... (Press Ctrl+C to stop)\n")
        
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
            
        try:
            while self.is_running:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
        except asyncio.CancelledError:
            pass
        finally:
            if self.audio_stream:
                self.audio_stream.close()

    async def send_realtime(self):
        """Send audio data to Gemini API"""
        try:
            while self.is_running:
                msg = await self.out_queue.get()
                await self.session.send(input=msg)
        except asyncio.CancelledError:
            pass

    async def receive_transcription(self):
        """Receive and display transcription from Gemini API"""
        try:
            while self.is_running:
                turn = self.session.receive()
                async for response in turn:
                    if text := response.text:
                        # Print the new text and update the full transcription
                        print(text, end="", flush=True)
                        self.transcription += text
        except asyncio.CancelledError:
            pass

    async def check_for_exit(self):
        """Check for user exit command"""
        try:
            while self.is_running:
                await asyncio.sleep(0.1)  # Small delay to prevent CPU hogging
        except asyncio.CancelledError:
            self.is_running = False
        except KeyboardInterrupt:
            self.is_running = False

    async def run(self):
        """Main execution loop"""
        print("\n📝 Live Audio Transcription")
        print("==========================")
        print("This app will transcribe your speech in real-time.")
        
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                self.out_queue = asyncio.Queue(maxsize=5)
                
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.receive_transcription())
                    tg.create_task(self.check_for_exit())
                    
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.close()
            
            print("\n\n📄 Final Transcription:")
            print("=======================")
            print(self.transcription or "No transcription generated.")
            print("\n✅ Transcription complete.")


if __name__ == "__main__":
    try:
        transcription = TranscriptionLoop()
        asyncio.run(transcription.run())
    except KeyboardInterrupt:
        print("\n\n🛑 Transcription stopped by user.")
    finally:
        # Clean up PyAudio
        pya.terminate()
