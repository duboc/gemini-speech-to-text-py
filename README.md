# Audio Transcription with Gemini API

This project provides three implementations for transcribing audio from your microphone using Google's Gemini API:

1. **Chunk-Based Transcription** (`transcribe_chunked.py`): Records audio in fixed-duration chunks, processes each chunk separately.
2. **Continuous Transcription** (`transcribe_continuous.py`): Continuously records audio and processes overlapping chunks for a more seamless experience.
3. **Live Transcription** (`transcribe_live.py`): Records audio and processes with the Gemini Live Api. 

**Note:** Both implementations use the standard Gemini API, which has higher latency compared to the real-time streaming Gemini Live API (see `transcribe_live.py` for reference).

## Features

- Real-time audio capture from your microphone
- Audio capture from your microphone
- Transcription using the standard Gemini API
- Simple command-line interface
- Two implementation options:
  - Chunk-based: Records and processes discrete chunks
  - Continuous: Records continuously with overlapping chunks

## Requirements

- Python 3.10 or higher
- A Gemini API key (stored in `.env` file)
- Required Python packages (listed in `requirements.txt`)

## Setup

1. Make sure you have Python installed on your system.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure your Gemini API key is in the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Chunk-Based Transcription
```
python transcribe_chunked.py
```
- Records audio in fixed-duration chunks (default 5 seconds)
- Processes each chunk separately
- Displays transcription for each chunk
- Press `Ctrl+C` to stop

### Continuous Transcription
```
python transcribe_continuous.py
```
- Records audio continuously
- Processes overlapping chunks for a more seamless experience
- Displays recent transcriptions with timestamps
- Press `Ctrl+C` to stop

## Files

- `transcribe_continuous.py`: Continuous transcription implementation (recommended)
- `transcribe_chunked.py`: Chunk-based transcription implementation
- `transcribe_live.py`: Live API implementation (for reference)
- `requirements.txt`: List of required Python packages
- `.env`: Contains your Gemini API key

## Troubleshooting

- If you encounter audio device errors, check your microphone settings
- Make sure your API key is correctly set in the `.env` file
- Ensure you have a stable internet connection for API communication
