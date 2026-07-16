# EchoVerse — Live AI Conversation tool

EchoVerse is a sleek interface for voice-based AI interactions. It allows users to record their voice, send the audio to a local backend, and view real-time streaming responses from an AI

## Features

* **Voice Recording:** Uses `RecordRTC` to capture high-quality stereo audio directly from the browser.
* **Live Streaming Responses:** Handles chunked text streaming from the backend, displaying the user's transcribed text and the AI's response in real-time.
* **Session Controls:** Options to discard a current recording, stop an ongoing AI response, copy the latest reply, or clear the conversation history.
* **Responsive Dark UI:** A mobile-friendly, modern design with status indicators, typing animations, and intuitive chat bubbles. Now features a dynamic light mode toggle and an animated ambient starfield background.
* **Continuous Live Mode:** Includes a custom `VoiceActivityDetector` for hands-free, continuous conversation without needing to repeatedly click the record button.
* **Real-Time Speech Synthesis:** Streams and plays synthesized audio gaplessly using Kokoro TTS on the backend and a custom Web Audio API player on the frontend.
* **Document Context (RAG):** Users can upload documents (PDF, DOCX, TXT, CSV, JSON) via a drag-and-drop zone to restrict the AI's knowledge strictly to that file.

## Prerequisites

Because this project is purely a frontend interface, it requires a companion backend server to function.

* A modern web browser with microphone permissions enabled.
* A local backend server 
* Python installed locally with dependencies for the backend (e.g., PyTorch, Transformers, faster-whisper, onnxruntime).
* A Hugging Face token saved in your local `.env` file to authenticate model downloads.

## Installation and Setup

1. **Download the project:** Save the provided code as `index.html`.
2. **Serve the file:** Because the application requests microphone access and makes API calls, it is best run through a local web server rather than opening the file directly (to avoid CORS or permission issues).
   * If you have Python installed, you can run:
     ```bash
     python -m http.server 8080
     ```
   * Or use the **Live Server** extension in VS Code.
3. **Open in Browser:** Navigate to `http://localhost:8080` (or whichever port your local server is using).
4. **Start the Backend:** Install the Python requirements and run the backend script:
   ```bash
   python newserver.py
   ```

## Backend API Specification

To make EchoVerse work, your local backend must strictly adhere to the following API contract.

### Endpoints

* `POST http://127.0.0.1:5000/api/chat` — The main conversational endpoint.
* `POST http://127.0.0.1:5000/api/document` — Uploads a document and extracts up to 6,000 characters for the AI's context.
* `DELETE http://127.0.0.1:5000/api/document` — Clears the current document context.
* `POST http://127.0.0.1:5000/api/clear` — Clears the conversation history (max 12 turns kept) and resets the KV-cache.

### Request Format (`multipart/form-data`)

* **`audio`**: The recorded audio file (WAV format, 16kHz, single channel).
* **`language`**: A string representing the selected language (e.g., "English", "Hindi", "French").

### Expected Response Format

The frontend expects a **streaming response** (chunked transfer encoding).

#### Legacy Format

The stream must output the user's transcribed text first, followed by a unique separator `±`, and then the AI's response.

> **Note:** If the server returns standard JSON with an `error` key, the frontend will display it as an error banner.

#### Binary Streaming Protocol

The system now utilizes a highly optimized binary stream (`application/octet-stream`) rather than raw text.

The stream sends chunks using the following byte structure:

```
[Type (1 byte)] [Length (4 bytes)] [Payload]
```

##### Frame Types

* `0x01` — User Transcript (Text)
* `0x02` — AI Response Chunk (Text)
* `0x03` — Synthesized Audio Chunk (WAV format)
* `0x04` — Split Marker (Indicates transition from User text to AI text)
* `0x05` — Empty/Hallucination (Instructs frontend to discard the turn)

## Technologies Used

* **HTML5 / CSS3:** Layout, styling, and CSS animations.
* **Vanilla JavaScript:** DOM manipulation, event handling, and Fetch API streams.
* **RecordRTC:** Third-party library (loaded via CDN) for reliable cross-browser audio recording.
* **Backend / Machine Learning:** Python, Flask, PyTorch, faster-whisper, and Kokoro TTS via ONNX.
* **STT used:** Whisper (Faster).
* **TTS used:** Kokoro.
* **Gemma 4 mode:** E2B.

## License

This project is open-source and available under the MIT License.
