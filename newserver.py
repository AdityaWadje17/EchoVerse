import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
import io
import torch
import soundfile as sf
from threading import Thread
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, TextIteratorStreamer

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Operational hardware detected: {device.upper()}")

# =====================================================================
# 1. LOAD WHISPER
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
# 2. LOAD GEMMA 4
# =====================================================================
print("Loading Gemma 4 E2B locally...")
gemma_id = "google/gemma-4-e2b-it"
gemma_processor = AutoProcessor.from_pretrained(gemma_id)
gemma_model = AutoModelForCausalLM.from_pretrained(
    gemma_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)

print("\nALL MODELS SUCCESSFULLY LOCKED TO GPU. LIVE STREAMING PIPELINE OPERATIONAL.")

# =====================================================================
# CONVERSATION MEMORY
# =====================================================================
# Simple in-memory history for a single ongoing conversation.
# Each entry: {"role": "user" | "assistant", "content": [{"type": "text", "text": "..."}]}
# This is intentionally process-global (not per-session) since this is a
# local single-user app. If you ever serve multiple users at once, key
# this dict by a session/user id instead.
conversation_history = []

# Cap how many past turns we feed back into the model so the prompt
# doesn't grow unbounded and slow down / blow past context length.
MAX_HISTORY_TURNS = 12  # a "turn" = one user message + one assistant reply


@app.route("/api/clear", methods=["POST"])
def clear_history():
    conversation_history.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/chat", methods=["POST"])
def voice_chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file detected"}), 400
        
    audio_file = request.files["audio"]
    # Language selection removed — always respond in English.
    target_language = "English"
    
    try:
        audio_bytes = audio_file.read()
        audio_data, sampling_rate = sf.read(io.BytesIO(audio_bytes))

        # =====================================================================
        # STEP 1: LISTEN (Whisper - ~0.5s delay)
        # =====================================================================
        input_features = whisper_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
        # Force English transcription — without this, Whisper auto-detects
        # the spoken language and can mis-guess it (e.g. transcribing
        # English speech as Bengali/Hindi gibberish).
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        
        transcribed_text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        print(f"\n[User Said]: {transcribed_text}")

        if not transcribed_text:
            return Response("I couldn't hear anything. Please try again.", mimetype='text/plain')

        # =====================================================================
        # STEP 2: PREPARE PROMPT (now with memory)
        # =====================================================================
        system_instruction = (
            "You are a helpful AI assistant having an ongoing spoken conversation "
            "with the user. Use the prior turns for context. Respond naturally and "
            "concisely, in English."
        )

        # Trim history to the last N turns (2 messages per turn: user + assistant)
        trimmed_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

        # Build the full message list: past turns (clean) + the new user turn
        # (system instruction is injected only on the newest message so it
        # doesn't get duplicated/stale across history).
        messages = list(trimmed_history) + [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_instruction}\n\nUser: {transcribed_text}"}
                ],
            }
        ]

        prompt = gemma_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
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

        # =====================================================================
        # STEP 3: LIVE STREAMING GENERATION (WITH SPLITTER)
        # =====================================================================
        streamer = TextIteratorStreamer(gemma_processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**cleaned_inputs, streamer=streamer, max_new_tokens=256)

        thread = Thread(target=gemma_model.generate, kwargs=generation_kwargs)
        thread.start()

        def generate_stream():
            # Instantly fire the transcription and the splitter symbol first
            yield f"{transcribed_text}±"

            # Then stream Gemma's generated text, while also collecting the
            # full reply so we can save it to memory once streaming ends.
            full_reply = []
            for new_text in streamer:
                full_reply.append(new_text)
                yield new_text

            assistant_text = "".join(full_reply).strip()

            # Save this exchange to memory (clean text, no system prefix)
            # so future turns have real context.
            conversation_history.append({
                "role": "user",
                "content": [{"type": "text", "text": transcribed_text}],
            })
            conversation_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            })

        return Response(stream_with_context(generate_stream()), mimetype='text/plain')
        
    except Exception as e:
        print(f"\n Internal processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)