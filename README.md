# GRAMMER-GURU
### PIPELINE
```
🎤 Voice Sample (.wav)
     ↓
🧠 Deepgram STT → transcript
     ↓
🔍 spaCy preprocessing + rule-based grammar analysis
     ↓
🤖  LLM → grammar score + natural feedback
     ↓
📦 JSON Response (score, feedback, transcript)
```


### How To Run
```bash
conda create  -n grammer-guru --file requirements.txt
conda activate grammer-guru
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload
streamlit run frontend/main.py
docker compose up -d # Don't run in your local as you don't have docker installed
```

### Run LLAMA.CPP without docker
```bash
brew install cmake # for MAC OS
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
LLAMA_METAL=1 make -j
./server -m /path/to/model.gguf --port 8080 --threads 8
```