import streamlit as st
import requests
import io


#  Backend Configuration

BACKEND_URL = "http://localhost:8000/analyze"  


#  Streamlit UI

st.set_page_config(page_title="Grammar Scoring Engine")

st.title(" Grammar Scoring Engine (Voice → Grammar Feedback)")
st.write("Upload a voice sample — we’ll transcribe it, clean it, and evaluate its grammar using Deepgram + spaCy + Qwen3 LLaMA.")

uploaded_file = st.file_uploader("🎤 Upload your audio file", type=["wav", "mp3", "m4a", "webm", "ogg"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Analyze Grammar"):
        with st.spinner("Processing your audio... please wait ⏳"):
            try:
                # Sending file to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.subheader(" Original Transcript")
                    st.text_area("Original Transcript", result["original_transcript"], height=150)

                    st.subheader(" Cleaned Transcript")
                    st.text_area("Cleaned Transcript", result["cleaned_transcript"], height=150)

                    st.subheader(" Grammar Feedback")
                    st.markdown(f"**Score:** {result['grammar_score']} / 10")
                    st.info(result["feedback"])
                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
