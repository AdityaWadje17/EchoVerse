import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # set this in your environment, not in code
import io
import re
import torch
import soundfile as sf
from threading import Thread
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, TextIteratorStreamer

# Document text-extraction libraries (both already installed)
import pdfplumber
import docx as docx_lib

# Kokoro TTS — runs on CPU on purpose. It's ~82M params and fast enough
# on CPU that it isn't worth spending your 8GB VRAM budget on, which
# Whisper + Gemma already use fully.
from kokoro_onnx import Kokoro

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Operational hardware detected: {device.upper()}")

# =====================================================================
# 1. LOAD WHISPER (GPU)
# =====================================================================
print("Loading Whisper-Small engine...")
whisper_id = "openai/whisper-small"
whisper_processor = AutoProcessor.from_pretrained(whisper_id)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

# =====================================================================
# 2. LOAD GEMMA 4 (bf16, no quantization — loads straight to GPU)
# =====================================================================
print("Loading Gemma 4 E2B locally...")
gemma_id = "google/gemma-4-e2b-it"

# NEVER hardcode tokens in source. Set HF_TOKEN as a real environment
# variable (e.g. in PowerShell: $env:HF_TOKEN="hf_xxx") before running.
hf_token = os.environ.get("HF_TOKEN") or None

gemma_processor = AutoProcessor.from_pretrained(gemma_id, token=hf_token)

if device == "cuda":
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"[Gemma] Free VRAM before load: {free_bytes / (1024 ** 3):.2f} GiB")

gemma_model = AutoModelForCausalLM.from_pretrained(
    gemma_id,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    token=hf_token,
).to(device)

# =====================================================================
# 3. LOAD KOKORO (CPU path-protected configuration)
# =====================================================================
import onnxruntime as ort

print("Loading Kokoro TTS...")
available_providers = ort.get_available_providers()
print(f"[Kokoro] Available ONNX Runtime providers: {available_providers}")

gpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
    if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
print(f"[Kokoro] Using providers: {gpu_providers}")

# Explicit absolute file path strings pointing right to your user account downloads directory
onnx_path = r"C:\Users\johne\Downloads\kokoro-v1.0.onnx"
voices_path = r"C:\Users\johne\Downloads\voices-v1.0.bin"

try:
    # Newer kokoro-onnx versions accept a providers kwarg directly.
    kokoro = Kokoro(onnx_path, voices_path, providers=gpu_providers)
except TypeError:
    # Older versions don't expose it — onnxruntime-gpu still auto-selects
    # CUDA on its own as long as it (not plain onnxruntime) is installed.
    kokoro = Kokoro(onnx_path, voices_path)

if "CUDAExecutionProvider" not in available_providers:
    print("[Kokoro] WARNING: CUDAExecutionProvider not available — running on CPU. "
          "Install onnxruntime-gpu (and uninstall plain onnxruntime) to enable GPU.")

DEFAULT_VOICE = "af_heart"   # swap for any voice in kokoro.get_voices()

# One-time warm-up: the first TTS call pays a fixed cost (espeak-ng phonemizer
# backend init, ONNX session first-run overhead). Paying it here at startup
# means your first real user request isn't the slow one.
print("[Kokoro] Warming up TTS engine...")
try:
    _ = kokoro.create("Warming up.", voice=DEFAULT_VOICE, speed=1.0, lang="en-us")
    print("[Kokoro] Warm-up complete.")
except Exception as e:
    print(f"[Kokoro] Warm-up failed (non-fatal): {e}")

print("\nALL MODELS SUCCESSFULLY LOADED. LIVE STREAMING PIPELINE OPERATIONAL.")

# =====================================================================
# CONVERSATION MEMORY
# =====================================================================
conversation_history = []

# =====================================================================
# DOCUMENT CONTEXT
# =====================================================================
document_context = {"text": "", "filename": ""}
MAX_DOC_CHARS = 6000


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)

    elif ext in ("docx", "doc"):
        doc = docx_lib.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext in ("txt", "md", "csv", "json", "py", "js", "html", "xml"):
        return file_bytes.decode("utf-8", errors="replace")

    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# =====================================================================
# DOCUMENT ENDPOINTS
# =====================================================================

@app.route("/api/document", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    filename = f.filename or "document"

    try:
        file_bytes = f.read()
        raw_text = extract_text(file_bytes, filename)

        if not raw_text.strip():
            return jsonify({"error": "Could not extract any text from this file."}), 400

        if len(raw_text) > MAX_DOC_CHARS:
            raw_text = raw_text[:MAX_DOC_CHARS] + "\n\n[Document truncated — too long to fit in context]"

        document_context["text"] = raw_text
        document_context["filename"] = filename

        word_count = len(raw_text.split())
        print(f"[Document] Loaded '{filename}' — {word_count} words, {len(raw_text)} chars")
        return jsonify({"status": "loaded", "filename": filename, "word_count": word_count})

    except Exception as e:
        print(f"[Document] Extraction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/document", methods=["DELETE"])
def remove_document():
    document_context["text"] = ""
    document_context["filename"] = ""
    print("[Document] Document context cleared.")
    return jsonify({"status": "cleared"})


# =====================================================================
# CLEAR CONVERSATION
# =====================================================================

@app.route("/api/clear", methods=["POST"])
def clear_history():
    conversation_history.clear()
    print("[Memory] Conversation history cleared.")
    return jsonify({"status": "cleared"})


# =====================================================================
# TTS ENDPOINT
# =====================================================================

def split_into_sentences(text: str):
    """Crude but effective sentence splitter so we can synthesize and stream
    audio sentence-by-sentence instead of waiting for the whole reply."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sentences if sentences else [text]


@app.route("/api/tts", methods=["POST"])
def synthesize_speech():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or DEFAULT_VOICE

    if not text:
        return jsonify({"error": "No text provided"}), 400

    def generate_audio_stream():
        for sentence in split_into_sentences(text):
            try:
                samples, sample_rate = kokoro.create(
                    sentence,
                    voice=voice,
                    speed=1.0,
                    lang="en-us"
                )
                buf = io.BytesIO()
                sf.write(buf, samples, sample_rate, format="WAV")
                buf.seek(0)
                yield buf.read()
            except Exception as e:
                print(f"[TTS] Sentence synthesis error: {e}")

    return Response(stream_with_context(generate_audio_stream()), mimetype="audio/wav")


# =====================================================================
# MAIN CHAT ENDPOINT
# =====================================================================

@app.route("/api/chat", methods=["POST"])
def voice_chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file detected"}), 400

    audio_file = request.files["audio"]

    try:
        audio_bytes = audio_file.read()
        audio_data, sampling_rate = sf.read(io.BytesIO(audio_bytes))

        # -----------------------------------------------------------------
        # STEP 1: TRANSCRIBE (Whisper, forced English)
        # -----------------------------------------------------------------
        input_features = whisper_processor(
            audio_data, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)

        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                predicted_ids = whisper_model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids
                )

        transcribed_text = whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()
        print(f"\n[User Said]: {transcribed_text}")

        if not transcribed_text:
            return Response("I couldn't hear anything. Please try again.", mimetype="text/plain")

        # -----------------------------------------------------------------
        # STEP 2: SAVE USER TURN TO HISTORY
        # -----------------------------------------------------------------
        conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": transcribed_text}],
        })

        # -----------------------------------------------------------------
        # STEP 3: BUILD PROMPT
        # -----------------------------------------------------------------
        doc_section = ""
        if document_context["text"]:
            doc_section = (
                f"\n\nYou have been given a document called '{document_context['filename']}'. "
                f"Answer the user's questions using only the facts contained in this document — "
                f"do not pull in outside facts, outside knowledge of the world, or information about "
                f"other people/companies/events not mentioned in the document. "
                f"However, you SHOULD reason over, summarize, compare, and draw conclusions from the "
                f"facts that ARE in the document — for example, inferring what roles someone is qualified "
                f"for based on the skills and experience listed, or identifying patterns/gaps. "
                f"This kind of reasoning over the document's own content is expected and encouraged. "
                f"Only refuse if the answer would require a fact that is simply not present anywhere in "
                f"the document (e.g. asking about something never mentioned at all) — in that case say: "
                f"'I can only answer questions based on the uploaded document, and this information is not in it.'\n\n"
                f"--- DOCUMENT START ---\n"
                f"{document_context['text']}\n"
                f"--- DOCUMENT END ---"
            )

        if document_context["text"]:
            system_instruction = (
                "You are a document assistant with strong analytical ability. "
                "You must not use outside/general knowledge to supply facts the document doesn't contain, "
                "but you should freely reason, infer, and draw conclusions using the facts that are in the document. "
                "Keep answers concise — 3 to 4 sentences maximum, no bullet points or headers, unless the user "
                "explicitly asks for a detailed or long-form answer."
                + doc_section
            )
        else:
            system_instruction = (
                "You are a helpful AI assistant having an ongoing spoken conversation "
                "with the user. You have full memory of everything said in this session. "
                "Use the complete conversation history for context. "
                "Respond naturally and concisely, in English."
            )

        messages = []
        for i, entry in enumerate(conversation_history):
            if i == 0:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"{system_instruction}\n\n{entry['content'][0]['text']}"
                    }]
                })
            else:
                messages.append(entry)

        prompt = gemma_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = gemma_processor(text=prompt, return_tensors="pt")

        cleaned_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if torch.is_floating_point(value) and device == "cuda":
                    cleaned_inputs[key] = value.to(device=device, dtype=torch.bfloat16)
                else:
                    cleaned_inputs[key] = value.to(device=device)
            else:
                cleaned_inputs[key] = value

        # -----------------------------------------------------------------
        # STEP 4: STREAM GENERATION
        # -----------------------------------------------------------------
        streamer = TextIteratorStreamer(
            gemma_processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(**cleaned_inputs, streamer=streamer, max_new_tokens=256)

        thread = Thread(target=gemma_model.generate, kwargs=generation_kwargs)
        thread.start()

        def generate_stream():
            yield f"{transcribed_text}±"

            full_reply = []
            for new_text in streamer:
                full_reply.append(new_text)
                yield new_text

            thread.join()

            assistant_text = "".join(full_reply).strip()
            conversation_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            })

        return Response(stream_with_context(generate_stream()), mimetype="text/plain")

    except Exception as e:
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
            print("[Memory] Rolled back orphaned user turn due to error.")
        print(f"\nInternal processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, threaded=True)