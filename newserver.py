import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

import io
import re
import time
import struct
import queue
import threading
import torch
import numpy as np
import soundfile as sf
from threading import Thread
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer

import pdfplumber
import docx as docx_lib
import onnxruntime as ort
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel

# Enable TF32 and cuDNN autotuning for faster matrix operations without loss of accuracy.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Binary frame type identifiers used when streaming responses to the client.
FRAME_TRANSCRIPT = 0x01
FRAME_AI_TEXT    = 0x02
FRAME_AUDIO      = 0x03
FRAME_SPLIT      = 0x04
FRAME_EMPTY      = 0x05

def pack_frame(ftype: int, data) -> bytes:
    """Encode a frame type and payload into a length-prefixed binary frame."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return struct.pack(">BI", ftype, len(data)) + data


def is_hallucination(text: str) -> bool:
    """Detect common Whisper hallucination patterns in transcribed text."""
    if not text or not text.strip():
        return True
    t = text.strip()
    KNOWN = {".", "..", "...", " ", "you", "thank you", "thanks",
             "thanks for watching", "please subscribe", "bye", "bye bye"}
    if t.lower() in KNOWN:
        return True
    if re.search(r'(.)\1{9,}', t):
        return True
    words = t.split()
    if len(words) >= 6 and words.count(words[0]) > len(words) * 0.65:
        return True
    if len(t) < 2:
        return True
    return False


app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Operational hardware detected: {device.upper()}")

# Speech recognition engine (faster-whisper / CTranslate2 backend).
print("Loading faster-whisper engine...")

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
whisper_compute_type = "float16" if device == "cuda" else "int8"

whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=device,
    compute_type=whisper_compute_type,
    download_root=os.environ.get("WHISPER_CACHE_DIR"),
)
print(f"[Whisper] faster-whisper '{WHISPER_MODEL_SIZE}' loaded on {device} "
      f"(compute_type={whisper_compute_type})")

# Language model used for conversational responses.
print("Loading Gemma 4 E2B locally...")
GEMMA_SNAPSHOT = r"C:\Users\tejas\.cache\huggingface\hub\models--google--gemma-4-E2B-it\snapshots\70af34e20bd4b7a91f0de6b22675850c43922a03"

gemma_processor = AutoProcessor.from_pretrained(GEMMA_SNAPSHOT, local_files_only=True)
gemma_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_SNAPSHOT,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    local_files_only=True,
    attn_implementation="sdpa",
).to(device)
gemma_model.eval()

# Text-to-speech engine.
print("Loading Kokoro TTS...")
available_providers = ort.get_available_providers()

if "DmlExecutionProvider" in available_providers:
    gpu_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
elif "CUDAExecutionProvider" in available_providers:
    gpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    gpu_providers = ["CPUExecutionProvider"]

print(f"[Kokoro] Using providers: {gpu_providers}")

onnx_path   = r"C:\Users\tejas\Downloads\kokoro-v1.0.onnx"
voices_path = r"C:\Users\tejas\Downloads\voices-v1.0.bin"

try:
    kokoro = Kokoro(onnx_path, voices_path, providers=gpu_providers)
except TypeError:
    kokoro = Kokoro(onnx_path, voices_path)

best_provider = gpu_providers[0]
if best_provider != "CPUExecutionProvider":
    try:
        kokoro.session = ort.InferenceSession(onnx_path, providers=gpu_providers)
        print(f"[Kokoro] Session patched to {best_provider}")
    except Exception as e:
        print(f"[Kokoro] GPU session patch failed, staying on CPU: {e}")

DEFAULT_VOICE = "af_heart"

# Concurrency control for Kokoro synthesis. Most onnxruntime sessions support
# concurrent Run() calls since session state is read-only after load, so no
# lock is used by default. Set KOKORO_SERIALIZE=1 to force sequential
# synthesis if issues are observed on a particular onnxruntime build.
KOKORO_SERIALIZE = os.environ.get("KOKORO_SERIALIZE", "0") == "1"
_kokoro_lock = threading.Lock() if KOKORO_SERIALIZE else None

print("[Kokoro] Warming up TTS engine...")
_t0 = time.time()
try:
    _ = kokoro.create("Warming up.", voice=DEFAULT_VOICE, speed=1.0, lang="en-us")
    _elapsed = time.time() - _t0
    _backend = "GPU" if _elapsed < 1.0 else f"CPU, synthesis is slow ({_elapsed:.1f}s per sentence)"
    print(f"[Kokoro] Warm-up complete in {_elapsed:.2f}s, backend: {_backend}")
except Exception as e:
    print(f"[Kokoro] Warm-up failed (non-fatal): {e}")

print("\nALL SYSTEMS READY.")

# Session state: conversation memory and uploaded document context.
conversation_history = []
document_context     = {"text": "", "filename": ""}
MAX_DOC_CHARS         = 6000

# Maximum number of turns retained in history to bound prompt growth over
# the course of a long-running session.
MAX_HISTORY_TURNS = 12

# Serializes GPU-bound model calls. Whisper and Gemma share a single GPU
# context, so an in-flight generation and an overlapping transcription
# (e.g. triggered by barge-in) must not run concurrently.
gpu_lock = threading.Lock()

# Lightweight KV-cache reuse: retains the last past_key_values along with
# the exact input_ids used to produce them. If a new turn's prompt is the
# same prefix plus a new suffix, only the suffix tokens are forwarded
# through the model. Falls back to a full prefill whenever the prefix does
# not match, for example after a document change or history trim.
_kv_cache_state = {"past_key_values": None, "input_ids": None}
_kv_lock = threading.Lock()


def synth_sentence(sentence: str, out_q: queue.Queue):
    """Synthesize a single sentence with Kokoro and push the resulting WAV bytes to out_q."""
    try:
        if _kokoro_lock is not None:
            with _kokoro_lock:
                samples, sr = kokoro.create(
                    sentence.strip(), voice=DEFAULT_VOICE, speed=1.0, lang="en-us"
                )
        else:
            samples, sr = kokoro.create(
                sentence.strip(), voice=DEFAULT_VOICE, speed=1.0, lang="en-us"
            )
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        out_q.put(buf.getvalue())
    except Exception as e:
        print(f"[TTS inline] Error: {e}")
        out_q.put(None)


def transcribe_audio(audio_data: np.ndarray, sampling_rate: int) -> str:
    """Transcribe mono float32 audio using faster-whisper."""
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)

    segments, _info = whisper_model.transcribe(
        audio_data,
        language="en",
        task="transcribe",
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
    )
    return "".join(seg.text for seg in segments).strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from an uploaded document based on its file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    elif ext in ("docx", "doc"):
        doc = docx_lib.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext in ("txt", "md", "csv", "json", "py", "js", "html", "xml"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def _reset_kv_cache():
    """Invalidate the cached KV state, forcing a full prefill on the next generation."""
    with _kv_lock:
        _kv_cache_state["past_key_values"] = None
        _kv_cache_state["input_ids"] = None


# Document management endpoints.
@app.route("/api/document", methods=["POST"])
def upload_document():
    """Accept an uploaded document, extract its text, and store it as active context."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    filename = f.filename or "document"
    try:
        raw = extract_text(f.read(), filename)
        if not raw.strip():
            return jsonify({"error": "Could not extract text."}), 400
        if len(raw) > MAX_DOC_CHARS:
            raw = raw[:MAX_DOC_CHARS] + "\n\n[Document truncated]"
        document_context["text"]     = raw
        document_context["filename"] = filename
        wc = len(raw.split())
        print(f"[Document] Loaded '{filename}', {wc} words")
        _reset_kv_cache()
        return jsonify({"status": "loaded", "filename": filename, "word_count": wc})
    except Exception as e:
        print(f"[Document] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/document", methods=["DELETE"])
def remove_document():
    """Clear the currently active document context."""
    document_context["text"] = document_context["filename"] = ""
    _reset_kv_cache()
    return jsonify({"status": "cleared"})

@app.route("/api/clear", methods=["POST"])
def clear_history():
    """Clear the stored conversation history."""
    conversation_history.clear()
    _reset_kv_cache()
    print("[Memory] Cleared.")
    return jsonify({"status": "cleared"})


# Main conversational endpoint: accepts audio, transcribes it, generates a
# response, and streams back transcript, text, and synthesized audio frames.
@app.route("/api/chat", methods=["POST"])
def voice_chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    try:
        audio_data, sampling_rate = sf.read(io.BytesIO(audio_file.read()))

        # Transcription is serialized against Gemma generation. faster-whisper
        # manages its own execution internally, but it shares the same
        # physical GPU as Gemma, so overlapping kernels from a barge-in
        # transcription mid-generation must be avoided.
        with gpu_lock:
            transcribed_text = transcribe_audio(audio_data, sampling_rate)
        print(f"\n[User Said]: {transcribed_text}")

        if is_hallucination(transcribed_text):
            print("[Whisper] Hallucination detected, ignoring.")
            return Response(pack_frame(FRAME_EMPTY, b""), mimetype="application/octet-stream")

        # Record the user turn.
        conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": transcribed_text}],
        })

        # Trim history so the prompt does not grow unbounded.
        if len(conversation_history) > MAX_HISTORY_TURNS:
            trimmed = len(conversation_history) - MAX_HISTORY_TURNS
            del conversation_history[:trimmed]
            _reset_kv_cache()
            print(f"[Memory] Trimmed {trimmed} old turn(s), kept last {MAX_HISTORY_TURNS}")

        print(f"[Memory] {len(conversation_history)} messages")

        # Build the system instruction and prompt, depending on whether a
        # document is currently active.
        if document_context["text"]:
            doc_block = (
                f"\n\nYou have been given a document called '{document_context['filename']}'. "
                f"Answer ONLY from this document. Do not use outside knowledge. "
                f"If the answer is not in the document say: "
                f"'I can only answer questions based on the uploaded document, and this information is not in it.'\n\n"
                f"--- DOCUMENT START ---\n{document_context['text']}\n--- DOCUMENT END ---"
            )
            system_instruction = (
                "You are a strict document assistant. No general knowledge, "
                "only the document below." + doc_block
            )
        else:
            system_instruction = (
                "You are a helpful AI assistant. You have full memory of this session. "
                "Respond naturally and concisely in English."
            )

        messages = []
        for i, entry in enumerate(conversation_history):
            if i == 0:
                messages.append({"role": "user", "content": [{"type": "text",
                    "text": f"{system_instruction}\n\n{entry['content'][0]['text']}"}]})
            else:
                messages.append(entry)

        prompt = gemma_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = gemma_processor(text=prompt, return_tensors="pt")

        input_device = "cuda:0" if device == "cuda" else "cpu"
        cleaned = {}
        for k, v in inputs.items():
            cleaned[k] = v.to(input_device) if isinstance(v, torch.Tensor) else v
        if "attention_mask" not in cleaned:
            cleaned["attention_mask"] = torch.ones_like(cleaned["input_ids"]).to(input_device)

        # Attempt to reuse the KV cache from the previous turn. This is only
        # valid when the current input_ids begin with exactly the same
        # tokens as the cached prefix, meaning the document and history
        # prefix are unchanged and only new tokens were appended.
        generation_kwargs = dict(**cleaned, max_new_tokens=256)
        with _kv_lock:
            prev_ids = _kv_cache_state["input_ids"]
            prev_pkv = _kv_cache_state["past_key_values"]
            cur_ids  = cleaned["input_ids"]
            if (
                prev_pkv is not None
                and prev_ids is not None
                and cur_ids.shape[1] > prev_ids.shape[1]
                and torch.equal(cur_ids[:, :prev_ids.shape[1]], prev_ids)
            ):
                # Feed only the new suffix tokens and reuse the cached prefix compute.
                new_tokens = cur_ids[:, prev_ids.shape[1]:]
                generation_kwargs["input_ids"] = new_tokens
                generation_kwargs["past_key_values"] = prev_pkv
                generation_kwargs["attention_mask"] = torch.ones(
                    (cur_ids.shape[0], cur_ids.shape[1]), device=cur_ids.device, dtype=torch.long
                )
                print(f"[KV-cache] Reusing prefix ({prev_ids.shape[1]} tokens), "
                      f"prefill only {new_tokens.shape[1]} new tokens")
            else:
                print(f"[KV-cache] Full prefill ({cur_ids.shape[1]} tokens)")

        streamer = TextIteratorStreamer(
            gemma_processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs["streamer"] = generation_kwargs.get("streamer", streamer)
        generation_kwargs["use_cache"] = True
        generation_kwargs["return_dict_in_generate"] = False

        # The GPU lock is held for the entire generation so that no other
        # request, such as a barge-in transcription, can run concurrently.
        gpu_lock.acquire()

        gen_result_holder = {}

        def _run_generate():
            try:
                with torch.no_grad():
                    gen_result_holder["out"] = gemma_model.generate(**generation_kwargs)
            finally:
                gpu_lock.release()

        gen_thread = Thread(target=_run_generate)
        gen_thread.start()

        def generate_stream():
            yield pack_frame(FRAME_TRANSCRIPT, transcribed_text)
            yield pack_frame(FRAME_SPLIT, b"")

            full_reply    = []
            sentence_buf  = ""
            audio_q       = queue.Queue()
            synth_threads = []

            SENT_RE          = re.compile(r'(?<=[.!?,;:])(?=\s|$)')
            MIN_SYNTH_CHARS  = 12
            MAX_BUFFER_CHARS = 45

            def kick_synth(s):
                """Start synthesis for a sentence fragment if it meets the minimum length."""
                if len(s.strip()) < MIN_SYNTH_CHARS:
                    return
                t = threading.Thread(target=synth_sentence, args=(s, audio_q), daemon=True)
                t.start()
                synth_threads.append(t)

            for token in streamer:
                full_reply.append(token)
                yield pack_frame(FRAME_AI_TEXT, token)

                sentence_buf += token
                parts = SENT_RE.split(sentence_buf)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        kick_synth(sentence)
                    sentence_buf = parts[-1]
                elif len(sentence_buf) >= MAX_BUFFER_CHARS:
                    kick_synth(sentence_buf)
                    sentence_buf = ""

                while True:
                    try:
                        wav = audio_q.get_nowait()
                        if wav:
                            yield pack_frame(FRAME_AUDIO, wav)
                    except queue.Empty:
                        break

            if sentence_buf.strip():
                kick_synth(sentence_buf.strip())

            gen_thread.join()

            for t in synth_threads:
                t.join()
            while True:
                try:
                    wav = audio_q.get_nowait()
                    if wav:
                        yield pack_frame(FRAME_AUDIO, wav)
                except queue.Empty:
                    break

            # Persist the assistant turn to conversation history.
            assistant_text = "".join(full_reply).strip()
            conversation_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            })
            print(f"[Memory] {len(conversation_history)} messages")

            # Store the new past_key_values along with the input_ids that
            # produced them, so the next turn can potentially reuse this
            # prefix. With return_dict_in_generate=False, the generation
            # output carries no cache, so a full prefill will occur on the
            # next turn regardless. Enabling return_dict_in_generate=True
            # with a compatible generation config would allow this reuse
            # to take effect; left conservative here to avoid errors on
            # library versions that do not support it.
            out = gen_result_holder.get("out")
            if out is not None and hasattr(out, "past_key_values") and out.past_key_values is not None:
                with _kv_lock:
                    _kv_cache_state["past_key_values"] = out.past_key_values
                    _kv_cache_state["input_ids"] = out.sequences if hasattr(out, "sequences") else None

        return Response(
            stream_with_context(generate_stream()),
            mimetype="application/octet-stream"
        )

    except Exception as e:
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
            print("[Memory] Rolled back orphaned user turn.")
        print(f"\nInternal error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, threaded=True)