"""
Continuous Audio Transcription using Gemini API

This script continuously captures audio from your microphone and sends overlapping
chunks to the Gemini API for transcription, creating a seamless transcription experience.

## Setup

Install dependencies:
```
pip install -r requirements.txt
```
Ensure your API key is in .env:
```
GEMINI_API_KEY=your_api_key_here
```
"""

import asyncio
import pyaudio
import os
import io
import wave
import time
from collections import deque
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
SAMPLE_RATE = 16000  # Sample rate for audio recording
CHUNK_SIZE = 1024
CHUNK_DURATION = 5  # Duration of each audio chunk in seconds
OVERLAP_DURATION = 1  # Overlap between chunks in seconds (to avoid missing words at boundaries)

# Gemini model configuration
MODEL = "gemini-2.0-flash"  # Using model from documentation that supports audio input

# System instruction for the model
SYSTEM_INSTRUCTION = "act as a live transcriber, only write back what you hear. no explanation and the language is portuguese but could mix english words."


class ContinuousTranscriber:
    def __init__(self):
        self.frames_queue = asyncio.Queue()
        self.is_running = True
        self.audio_stream = None
        self.pya = pyaudio.PyAudio()
        self.client = genai.Client(api_key=api_key)
        
        # Buffer to store recent audio frames for creating overlapping chunks
        self.frame_buffer = deque(maxlen=int(SAMPLE_RATE / CHUNK_SIZE * (CHUNK_DURATION + OVERLAP_DURATION)))
        
        # Store recent transcriptions for display
        self.recent_transcriptions = []
        self.max_recent_transcriptions = 5

    async def record_audio(self):
        """Continuously record audio and add frames to the buffer and queue."""
        try:
            mic_info = self.pya.get_default_input_device_info()
            self.audio_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            
            print("\n🎤 Recording started. Press Ctrl+C to stop.")
            
            # Calculate how many chunks to collect before starting to process
            chunks_per_processing = int(SAMPLE_RATE / CHUNK_SIZE * OVERLAP_DURATION)
            chunk_counter = 0
            
            while self.is_running:
                try:
                    data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                    self.frame_buffer.append(data)
                    
                    # Every OVERLAP_DURATION seconds, signal that we have enough data to process
                    chunk_counter += 1
                    if chunk_counter >= chunks_per_processing:
                        # Put a marker in the queue to signal processing
                        await self.frames_queue.put("process")
                        chunk_counter = 0
                        
                except IOError as ex:
                    if getattr(ex, 'errno', None) != pyaudio.paInputOverflowed:
                        raise
                    print("Warning: Input overflowed.")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n❌ Error in audio recording: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()

    async def process_audio(self):
        """Process audio chunks from the queue and send to API."""
        try:
            while self.is_running:
                # Wait for a signal to process
                signal = await self.frames_queue.get()
                
                if signal == "process" and len(self.frame_buffer) >= int(SAMPLE_RATE / CHUNK_SIZE * CHUNK_DURATION):
                    # Create a copy of the current buffer for processing
                    frames_to_process = list(self.frame_buffer)[-int(SAMPLE_RATE / CHUNK_SIZE * CHUNK_DURATION):]
                    
                    # Convert frames to WAV format
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(self.pya.get_sample_size(FORMAT))
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(b''.join(frames_to_process))
                    wav_buffer.seek(0)
                    audio_bytes = wav_buffer.read()
                    
                    # Process in a separate task to avoid blocking
                    asyncio.create_task(self.transcribe_and_display(audio_bytes))
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n❌ Error in audio processing: {e}")

    async def transcribe_and_display(self, audio_bytes):
        """Send audio to API and display transcription."""
        try:
            # Prepare the content
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=SYSTEM_INSTRUCTION),
                        types.Part.from_bytes(data=audio_bytes, mime_type='audio/wav')
                    ],
                ),
            ]
            
            # Use the asynchronous API client
            response = await self.client.aio.models.generate_content(
                model=MODEL,
                contents=contents
            )
            
            # Extract text from response
            transcription = ""
            if response and hasattr(response, 'parts') and response.parts:
                transcription = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif response and hasattr(response, 'text'):
                transcription = response.text
                
            if transcription:
                # Add timestamp and store in recent transcriptions
                timestamp = time.strftime("%H:%M:%S")
                self.recent_transcriptions.append(f"[{timestamp}] {transcription}")
                
                # Keep only the most recent transcriptions
                if len(self.recent_transcriptions) > self.max_recent_transcriptions:
                    self.recent_transcriptions.pop(0)
                
                # Display all recent transcriptions
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\n🎙️  Continuous Transcription")
                print("==========================")
                print("Recent transcriptions (newest at bottom):")
                for t in self.recent_transcriptions:
                    print(t)
                print("\nRecording... Press Ctrl+C to stop.")
            
        except Exception as e:
            print(f"\n❌ Error during transcription: {e}")

    async def run(self):
        """Main execution method."""
        try:
            print("\n🎙️  Continuous Audio Transcription Started")
            print("=========================================")
            print("Recording and transcribing continuously...")
            
            # Start tasks
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.record_audio())
                tg.create_task(self.process_audio())
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
        finally:
            self.is_running = False
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            print("\n✅ Transcription process finished.")

async def main():
    transcriber = ContinuousTranscriber()
    await transcriber.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Transcription stopped by user.")
    finally:
        # Clean up PyAudio
        pyaudio.PyAudio().terminate()
