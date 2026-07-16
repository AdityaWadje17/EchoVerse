# EchoVerse: Future Development & Maintenance Roadmap

## Overview

This document outlines the planned enhancements, architectural improvements, and maintenance tasks for the EchoVerse project. It serves as a guide for future developers taking over the codebase to understand the project's trajectory and areas for optimization.

## High-Priority Enhancements (Immediate Next Steps)

- [ ] **Automatic Language Recognition:** Transition from the current manual language toggle—which supports English, Hindi, and French—to an automated detection system. This can be achieved by leveraging the Whisper engine's built-in language identification capabilities rather than hardcoding it.
- [ ] **Containerization (Docker):** Package the frontend and backend architectures into Docker containers. This will standardize the deployment process, properly manage the complex Python dependencies (such as `faster-whisper`, `torch`, and `onnxruntime`), and ensure environment consistency across different machines.
- [ ] **Latency Optimization:** Profile and further decrease the end-to-end latency of the application. Focus areas include tuning Whisper's `beam_size`, optimizing the text streaming pipeline using the `±` chunk separator, and refining the KV-cache reuse logic to prevent full prompt prefilling on every turn.

## Model & AI Pipeline Upgrades

- [ ] **Complete the RAG Architecture:** Finalize the ongoing development of the Retrieval-Augmented Generation (RAG) memory architecture to improve conversations regarding specific datasets. Currently, document parsing (PDF/DOCX) truncates inputs at 6,000 characters. Implementing a vector database will allow for proper document chunking and retrieval without data loss.
- [ ] **Evaluate 4-bit Quantization:** The Gemma 4 E2B model is currently configured to load in 8-bit precision (`load_in_8bit=True`). Evaluate migrating to 4-bit quantization (e.g., using NF4) to further reduce VRAM consumption and increase generation speed without significantly degrading response quality.
- [ ] **TTS Provider Stability:** Improve the initialization stability of the Kokoro TTS engine. Address edge cases where the ONNX Runtime session fails to initialize the GPU execution provider and falls back to slower CPU synthesis.

## Architecture & Scalability

- [ ] **Persistent Session Management:** The backend currently uses a basic in-memory list (`conversation_history`) to store active chats. Migrate this to a persistent caching layer or database (such as Redis or PostgreSQL) to support concurrent users, prevent memory leaks, and allow for long-term chat history retention.
- [ ] **Dynamic GPU Resource Management:** The application currently relies on a thread lock (`gpu_lock`) to serialize generation and transcription tasks, preventing overlapping CUDA kernels. Explore more advanced queueing mechanisms or separate worker nodes to handle simultaneous TTS and LLM inference more efficiently.

## Documentation & Testing

- [ ] **Automated Testing:** Implement unit tests for core backend functions, specifically targeting the hallucination detection logic (`is_hallucination`) and the binary frame packing utilities.
- [ ] **API Error Handling Enhancement:** Refine the frontend's error banner display logic to gracefully handle backend timeouts or model out-of-memory (OOM) failures while streaming chunked responses.