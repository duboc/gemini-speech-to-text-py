"""
Chunk-Based Audio Transcription using Standard Gemini API

This script captures audio from your microphone in chunks, sends each chunk
to the standard Google Gemini API for transcription, and displays the results.

Note: This approach has higher latency compared to the Gemini Live API.

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

# Gemini model configuration
MODEL = "gemini-2.0-flash-lite" # Using model from documentation that supports audio input

# Initialize PyAudio
pya = pyaudio.PyAudio()

# Configure Gemini client (as per example-lite.py)
client = genai.Client(api_key=api_key)

# System instruction for the model
SYSTEM_INSTRUCTION = "act as a live transcriber, only write back what you hear. no explanation and the language is portuguese but could mix english words."

def transcribe_audio_chunk(audio_bytes: bytes) -> str:
    """Sends an audio chunk (WAV format) to Gemini API and returns transcription."""
    try:
        # Prepare the content following the structure in example-lite.py
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=SYSTEM_INSTRUCTION),  # System instruction
                    # Try using from_bytes for audio data
                    types.Part.from_bytes(data=audio_bytes, mime_type='audio/wav')
                ],
            ),
        ]
        
        # Call generate_content on client.models, as shown in example-lite.py and documentation
        response = client.models.generate_content(
            model=MODEL,
            contents=contents
        )
        
        # Extract text from the response
        # Ensure response object is checked before accessing attributes
        if response and hasattr(response, 'parts') and response.parts:
             return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif response and hasattr(response, 'text'):
             return response.text # Handle cases where text is directly in response
        else:
             print(f"Warning: No text found in response object or its parts. Response: {response}")
             return ""

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        # print(f"Response received: {getattr(response, '_raw_response', 'N/A')}") # Uncomment for debugging
        return "[Transcription Error]"


async def main():
    """Main execution loop for chunk-based transcription."""
    stream = None
    try:
        mic_info = pya.get_default_input_device_info()
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        print("\n🎙️  Chunk-Based Transcription Started")
        print(f"Recording in {CHUNK_DURATION}-second chunks. Press Ctrl+C to stop.")
        print("==========================================")

        while True:
            frames = []
            print(f"\n[*] Recording {CHUNK_DURATION}-second chunk...")
            
            # Calculate number of chunks to read for the desired duration
            num_chunks_to_read = int(SAMPLE_RATE / CHUNK_SIZE * CHUNK_DURATION)
            
            for _ in range(num_chunks_to_read):
                try:
                    data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                except IOError as ex:
                    if ex[1] != pyaudio.paInputOverflowed:
                        raise
                    print("Warning: Input overflowed. Skipping chunk.")
                    # Skip this chunk if overflow occurs
                    frames = [] # Clear frames to avoid processing partial/corrupt data
                    break 
            
            if not frames: # If frames list is empty (due to overflow skip), continue to next iteration
                 continue

            print("[*] Recording finished. Transcribing...")

            # Convert raw audio frames to WAV format in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pya.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()

            # Transcribe the chunk (run synchronously for simplicity in the loop)
            # For better performance, this could be run in a separate thread/process
            # transcription = await asyncio.to_thread(transcribe_audio_chunk, audio_bytes)
            
            # Running synchronously as generate_content is blocking
            transcription = transcribe_audio_chunk(audio_bytes) 

            if transcription:
                print(f"Transcription: {transcription}")
            else:
                print("[No transcription received for this chunk]")

    except KeyboardInterrupt:
        print("\n\n🛑 Transcription stopped by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        print("\n✅ Transcription process finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass # Already handled in main
    finally:
        # Clean up PyAudio
        pya.terminate()
