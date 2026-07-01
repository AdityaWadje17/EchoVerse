# EchoVerse
# EchoVerse — Gemma 4 Voice

EchoVerse is a sleek, dark-themed web interface for voice-based AI interactions. It allows users to record their voice, send the audio to a local backend, and view real-time streaming responses from an AI (like Gemma 4). 

## Features

* **Voice Recording:** Uses `RecordRTC` to capture high-quality stereo audio directly from the browser.
* **Live Streaming Responses:** Handles chunked text streaming from the backend, displaying the user's transcribed text and the AI's response in real-time.
* **Language Selection:** Built-in toggle to switch between English, Hindi, and French.
* **Session Controls:** Options to discard a current recording, stop an ongoing AI response, copy the latest reply, or clear the conversation history.
* **Responsive Dark UI:** A mobile-friendly, modern design with status indicators, typing animations, and intuitive chat bubbles.

## Prerequisites

Because this project is purely a frontend interface, it requires a companion backend server to function. 

* A modern web browser with microphone permissions enabled.
* A local backend server running on `http://127.0.0.1:5000`.

## Installation and Setup

1. **Download the project:** Save the provided code as `index.html`.
2. **Serve the file:** Because the application requests microphone access and makes API calls, it is best run through a local web server rather than opening the file directly (to avoid CORS or permission issues).
   * If you have Python installed, you can run: `python -m http.server 8080`
   * Or use the "Live Server" extension in VS Code.
3. **Open in Browser:** Navigate to `http://localhost:8080` (or whichever port your local server is using).

## Backend API Specification

To make EchoVerse work, your local backend must strictly adhere to the following API contract:

### Endpoint
`POST http://127.0.0.1:5000/api/chat`

### Request Format (`multipart/form-data`)
* **`audio`**: The recorded audio file (WAV format, 16kHz, single channel).
* **`language`**: A string representing the selected language (e.g., "English", "Hindi", "French").

### Expected Response Format
The frontend expects a **streaming response** (chunked transfer encoding). 

The stream must output the user's transcribed text first, followed by a unique separator `±`, and then the AI's response. 

**Example Stream Sequence:**
1. `Hello, how are`
2. ` you?`
3. `±` *(This triggers the UI to create the AI chat bubble)*
4. `I am `
5. `doing great! How can I help?`

*Note: If the server returns standard JSON with an `error` key, the frontend will display it as an error banner.*

## Technologies Used

* **HTML5 / CSS3:** Layout, styling, and CSS animations.
* **Vanilla JavaScript:** DOM manipulation, event handling, and Fetch API streams.
* **RecordRTC:** Third-party library (loaded via CDN) for reliable cross-browser audio recording.

## License

This project is open-source and available under the MIT License.

## Ongoing Work

Currently I am working on building a RAG with memory architecture which will be added to this application to make it more usefull while conversing about some particular dataset