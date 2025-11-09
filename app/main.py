import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from deepgram import DeepgramClient
# from openai import OpenAI
import spacy
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import requests
import json 
import re

load_dotenv()

# -------------------------
# 🔧 Environment Setup
# -------------------------
DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

if not DG_API_KEY or not LLM_API_URL or not LLM_API_KEY :
    raise RuntimeError("Please set DEEPGRAM_API_KEY and OPENAI_API_KEY environment variables")

# ✅ Initialize clients
dg_client = DeepgramClient(api_key=DG_API_KEY)
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Grammar Scoring Engine (Voice → Grammar Feedback)")


# -------------------------
# 🧠 Helper Functions
# -------------------------


async def transcribe_audio(file_path: str, mimetype: str) -> str:
    """Transcribe local file (SDK v5.x)"""
    with open(file_path, "rb") as f:
        # v5 file transcription (NOTE: request is raw bytes)
        resp = dg_client.listen.v1.media.transcribe_file(
            request=f.read(),
            model="nova-3",          # or "nova-2" if you prefer
            smart_format=True,
            punctuate=True,
            # no need to pass mimetype here in v5; server infers it
        )

    # Support both object-style and dict-style responses
    try:
        return resp.results.channels[0].alternatives[0].transcript
    except AttributeError:
        return resp["results"]["channels"][0]["alternatives"][0]["transcript"]
    
def generate_feedback(text: str) -> dict:
    """Use self-hosted LLaMA.cpp (Qwen3) model to score and give feedback."""

    prompt = f"""
    You are an English grammar evaluator.

    Evaluate the grammar in the following transcript.
    Rate grammar correctness from 0 to 10 (10 = perfect grammar),
    and provide concise feedback in 2–3 sentences.

    Transcript:
    \"\"\"{text}\"\"\"

    Respond ONLY in valid JSON:
    {{
      "score": <number>,
      "feedback": "<your feedback text>"
    }}
    """

    try:
        response = requests.post(
            f"https://{LLM_API_URL}/v1/chat/completions",# https localhost:8000/v1/chat/completions
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-dev-qwen3"
            },
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # 🧹 Step 1: Extract JSON from response (ignore <think> blocks)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return {"score": None, "feedback": content.strip()}

        json_str = json_match.group(0)

        # 🧠 Step 2: Parse and clean
        parsed = json.loads(json_str)
        score = parsed.get("score")
        feedback = parsed.get("feedback", "").strip()

        return {"score": score, "feedback": feedback}

    except Exception as e:
        return {"score": None, "feedback": f"Error contacting model: {e}"}


def preprocess_text(text: str) -> str:
    """Use spaCy to clean text and remove fillers, preserving punctuation and spacing properly."""
    doc = nlp(text)
    fillers = {"um", "uh", "like", "you know", "hmm"}
    cleaned_sentences = []

    for sent in doc.sents:
        tokens = [t.text for t in sent if t.text.lower() not in fillers]
        sentence = " ".join(tokens)

        # --- 🧹 Fix common spacing issues ---
        sentence = re.sub(r"\s+([.,!?;:])", r"\1", sentence)  # remove space before punctuation
        sentence = re.sub(r"\s+'", "'", sentence)              # fix spaces before apostrophes
        sentence = re.sub(r"'\s+", "'", sentence)              # fix spaces after apostrophes
        sentence = re.sub(r'\s+"', '"', sentence)              # fix spaces before quotes
        sentence = re.sub(r'"\s+', '"', sentence)              # fix spaces after quotes
        sentence = sentence.strip()

        if sentence:
            cleaned_sentences.append(sentence)

    # Return clean, readable text
    cleaned_text = " ".join(cleaned_sentences)
    return cleaned_text


# -------------------------
# 🚀 API Endpoint
# -------------------------

@app.post("/analyze")
async def analyze_audio(file: UploadFile):
    """Upload an audio file and get grammar feedback"""
    allowed_exts = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext}. Allowed: {', '.join(sorted(allowed_exts))}"
        )

    # Infer mimetype (prefer UploadFile.content_type, fall back to extension)
    ct = (file.content_type or "").lower()
    ext_to_mt = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",   # sometimes audio/x-m4a; Deepgram accepts audio/mp4
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
    }
    mimetype = ct if ct.startswith("audio/") else ext_to_mt.get(ext, "application/octet-stream")

    # Save with the ORIGINAL extension
    with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Step 1: Transcribe
        transcript = await transcribe_audio(tmp_path, mimetype)

        # Step 2: Clean text
        cleaned_text = preprocess_text(transcript)

        # Step 3: Generate feedback
        feedback = generate_feedback(cleaned_text)

        return {
            "original_transcript": transcript,
            "cleaned_transcript": cleaned_text,
            "grammar_score": feedback.get("score"),
            "feedback": feedback.get("feedback"),
        }
    except Exception as e:
        # Surface Deepgram/SDK errors clearly
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
