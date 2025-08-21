# app.py — JioMart AI Shopping Co-Pilot (Voice + Regional Languages)
# Streamlit demo using Gemma 3 via ChatOllama
# ---------------------------------------------------------------
# Features:
# - Voice input in Indian languages (or type text)
# - Gemma 3 extracts structured cart JSON and replies in same language
# - Cart preview + inline editing + JSON export
# - Pluggable STT: Vosk (offline) if available, else Google SpeechRecognition
#
# NOTE:
# - Configure your Gemma 3 endpoint (Ollama/ChatOllama) in the "LLM SETTINGS" block.
# - Optional: set VOSK model paths per language to enable fully offline STT.

import json
import io
import os
from typing import List, Dict, Any

import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# --- Optional STT libs ---
# Install both; we'll try Vosk first, then fallback to Google SR
vosk_available = True
try:
    from vosk import Model as VoskModel, KaldiRecognizer
except Exception:
    vosk_available = False

sr_available = True
try:
    import speech_recognition as sr
except Exception:
    sr_available = False

# --- LLM (Gemma 3 via LangChain ChatOllama) ---
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


# =========================
# CONFIG: PAGE
# =========================
st.set_page_config(
    page_title="JioMart AI Shopping Co-Pilot",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 JioMart AI Shopping Co-Pilot")
st.caption("Voice + regional language understanding • Gemma-3 powered • Cart extraction & editing")


# =========================
# SIDEBAR: SETTINGS
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    # ---- LLM SETTINGS ----
    st.subheader("LLM")
    base_url = st.text_input("Gemma Base URL", value="http://10.166.189.167", help="Your ChatOllama base URL")
    model_name = st.text_input("Model Name", value="gemma3:12b", help="E.g., gemma3:12b")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    # ---- VOICE / LANGUAGE ----
    st.subheader("Voice & Language")
    lang_label_to_code = {
        "English (India)": "en-IN",
        "Hindi": "hi-IN",
        "Marathi": "mr-IN",
        "Tamil": "ta-IN",
        "Telugu": "te-IN",
        "Bengali": "bn-IN",
        "Kannada": "kn-IN",
        "Gujarati": "gu-IN",
        "Malayalam": "ml-IN",
        "Punjabi": "pa-IN",
        "Urdu": "ur-IN"
    }
    language_label = st.selectbox("Input Language", list(lang_label_to_code.keys()), index=1)
    language_code = lang_label_to_code[language_label]

    st.write("Speech-to-Text preference:")
    use_vosk_pref = st.radio("STT Engine", ["Auto (Vosk→Google)", "Vosk (Offline only)", "Google SR (Online)"], index=0)

    st.caption("💡 Tip: For fully offline demos, install Vosk and download a language model. Set its folder below.")
    vosk_model_root = st.text_input(
        "VOSK_MODEL_DIR (folder path)",
        value=os.getenv("VOSK_MODEL_DIR", ""),
        help="Folder containing subfolders per language, e.g., .../vosk-model-small-hi-0.22"
    )

    st.divider()
    st.markdown("**Export Options**")
    default_filename = st.text_input("Export filename", value="jiomart_cart.json")


# =========================
# LLM: PROMPT & PARSER
# =========================
cart_schema = ResponseSchema(
    name="cart",
    description=("List of items. Each item must have: "
                 "`product` (string), `quantity` (number), `unit` (string like kg/l/pcs), "
                 "`category` (e.g., vegetables, dairy, staples).")
)
reply_schema = ResponseSchema(
    name="reply",
    description="Conversational confirmation in the SAME language as the user's input."
)
schemas = [cart_schema, reply_schema]
output_parser = StructuredOutputParser.from_response_schemas(schemas)
FORMAT_INSTRUCTIONS = output_parser.get_format_instructions()

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
You are JioMart's AI Shopping Co-Pilot.
Your tasks:
1) Understand grocery requests in ANY Indian language (Hindi, Hinglish, Marathi, Tamil, Telugu, Bengali, etc).
2) Extract a structured cart (product, quantity, unit, category).
3) Reply back to the user in the SAME language they used.

Constraints:
- Use common Indian retail units (kg, g, l, ml, pcs, dozen).
- If quantity or unit is missing, infer the most common pack for that product and still return a best-guess.
- Keep product names concise (avoid extra descriptors).
- Keep categories generic (vegetables, fruits, dairy, bakery, beverages, snacks, personal care, home care, staples).

{format_instructions}

User said: ```{user_input}```
"""
)


# =========================
# HELPERS
# =========================
@st.cache_resource(show_spinner=False)
def load_vosk_model_for_language(lang_code: str, model_root: str):
    """
    Try to load a Vosk model appropriate for the language code.
    Heuristic mapping based on folder names in model_root.
    """
    if not vosk_available or not model_root or not os.path.isdir(model_root):
        return None

    # Simple heuristic: pick the first model dir under model_root
    # or require user to point to the specific language folder.
    if os.path.isdir(model_root):
        try:
            return VoskModel(model_root)
        except Exception:
            return None
    return None


def audio_bytes_to_pcm16_wav(audio_bytes: bytes) -> bytes:
    """
    Ensure audio is 16kHz mono PCM WAV for Vosk.
    streamlit_mic_recorder gives WAV already, but we normalize.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # mono, 16kHz, 16-bit
    out_buf = io.BytesIO()
    audio.export(out_buf, format="wav")
    return out_buf.getvalue()


def stt_transcribe(audio_bytes: bytes, lang_code: str, engine_pref: str, vosk_model_dir: str) -> str:
    """
    Try Vosk first (if chosen/available), else fallback to Google SpeechRecognition.
    """
    # Decide engine
    try_vosk = (engine_pref in ["Auto (Vosk→Google)", "Vosk (Offline only)"]) and vosk_available
    try_google = (engine_pref in ["Auto (Vosk→Google)", "Google SR (Online)"]) and sr_available

    # Attempt Vosk
    if try_vosk:
        model = load_vosk_model_for_language(lang_code, vosk_model_dir)
        if model is not None:
            try:
                wav_pcm16 = audio_bytes_to_pcm16_wav(audio_bytes)
                rec = KaldiRecognizer(model, 16000)
                rec.AcceptWaveform(wav_pcm16)
                result = json.loads(rec.Result())
                text = (result.get("text") or "").strip()
                if text:
                    return text
            except Exception as e:
                st.warning(f"Vosk STT failed, falling back (reason: {e})")

        elif engine_pref == "Vosk (Offline only)":
            st.error("Vosk model not found/loaded. Provide a valid VOSK_MODEL_DIR for offline STT.")
            return ""

    # Attempt Google SpeechRecognition
    if try_google:
        try:
            recognizer = sr.Recognizer()
            wav_buf = io.BytesIO(audio_bytes)
            with sr.AudioFile(wav_buf) as source:
                audio = recognizer.record(source)
            # Example: hi-IN, ta-IN, en-IN, etc.
            return recognizer.recognize_google(audio, language=lang_code)
        except Exception as e:
            st.error(f"Google SpeechRecognition failed: {e}")
            return ""

    st.error("No STT engine available. Install Vosk or SpeechRecognition.")
    return ""


@st.cache_resource(show_spinner=False)
def init_llm(_base_url: str, _model_name: str, _temperature: float):
    return ChatOllama(
        temperature=_temperature,
        model=_model_name,
        format="json",
        base_url=_base_url
    )


def call_llm(user_text: str, chat):
    prompt = PROMPT_TEMPLATE.format(
        user_input=user_text,
        format_instructions=FORMAT_INSTRUCTIONS
    )
    resp = chat.invoke(prompt)
    # Defensive parsing
    try:
        parsed = output_parser.parse(resp.content)
    except Exception:
        # As a fallback, try to json.loads the content
        try:
            parsed = json.loads(resp.content)
        except Exception:
            parsed = {"cart": [], "reply": "Sorry, I couldn't parse that."}
    # Normalize keys
    cart = parsed.get("cart", []) or []
    reply = parsed.get("reply", "") or ""
    # Enforce minimal structure
    norm_cart: List[Dict[str, Any]] = []
    for item in cart:
        norm_cart.append({
            "product": str(item.get("product", "")).strip(),
            "quantity": float(item.get("quantity", 1) or 1),
            "unit": str(item.get("unit", "pcs")).strip(),
            "category": str(item.get("category", "others")).strip()
        })
    return norm_cart, reply


# =========================
# MAIN UI
# =========================
# Session state
if "last_cart" not in st.session_state:
    st.session_state.last_cart = []
if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# Input area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎤 Speak your request")
    st.caption("Click **Record** → speak in your chosen language → click **Stop**.")

    audio = mic_recorder(
        start_prompt="Record",
        stop_prompt="Stop",
        just_once=False,
        key="recorder",
        format="wav",
        callback=None
    )

    if audio and audio.get("bytes"):
        st.audio(audio["bytes"], format="audio/wav")
        if st.button("Transcribe Voice", type="primary", use_container_width=True):
            text = stt_transcribe(audio["bytes"], language_code, use_vosk_pref, vosk_model_root)
            st.session_state.transcript = text
            if text:
                st.success("Transcription complete.")
            else:
                st.warning("No transcription produced. Try again or switch STT engine.")

with col2:
    st.subheader("⌨️ Or type your request")
    user_text_input = st.text_area(
        "Example: “मुझे 2 किलो आलू, 1 लीटर दूध और 1 पैकेट ब्रेड चाहिए”",
        value=st.session_state.transcript,
        height=120
    )
    st.caption("You can paste/edit the transcribed text here before sending to the AI.")

# Action buttons
run_cols = st.columns([1, 1, 1])
with run_cols[0]:
    run_clicked = st.button("🤖 Build Cart with AI", type="primary")
with run_cols[1]:
    clear_clicked = st.button("🧹 Clear")
with run_cols[2]:
    sample_clicked = st.button("✨ Fill Sample (Hindi)")

if sample_clicked:
    st.session_state.transcript = "मुझे 2 किलो आलू, 1 लीटर दूध, 6 अंडे और 1 पैकेट बिस्कुट चाहिए"
    st.experimental_rerun()

if clear_clicked:
    st.session_state.transcript = ""
    st.session_state.last_cart = []
    st.session_state.last_reply = ""
    st.experimental_rerun()

# Call LLM
if run_clicked:
    text_for_llm = user_text_input.strip()
    if not text_for_llm:
        st.warning("Please provide some text (via voice transcription or typing).")
    else:
        with st.spinner("Thinking with Gemma-3…"):
            chat = init_llm(base_url, model_name, temperature)
            cart, reply = call_llm(text_for_llm, chat)
            st.session_state.last_cart = cart
            st.session_state.last_reply = reply

# Output section
st.divider()
st.subheader("🛍️ Cart Preview & Editing")

if st.session_state.last_cart:
    # Use data_editor for inline editing
    edited = st.data_editor(
        st.session_state.last_cart,
        num_rows="dynamic",
        use_container_width=True,
        key="cart_editor"
    )
    st.session_state.last_cart = edited

    st.caption("Tip: You can add/remove rows, change quantities or units.")
else:
    st.info("Cart will appear here after you run the AI.")

# AI reply (in same language)
if st.session_state.last_reply:
    st.markdown("**🤖 Assistant says:**")
    st.write(st.session_state.last_reply)

# Export JSON
st.divider()
st.subheader("📤 Export / Integrate")

cart_json_str = json.dumps(
    {"cart": st.session_state.last_cart, "reply": st.session_state.last_reply},
    ensure_ascii=False,
    indent=2
)

st.code(cart_json_str, language="json")

col_dl, col_stub = st.columns([1, 1])

with col_dl:
    st.download_button(
        label="⬇️ Download cart JSON",
        data=cart_json_str.encode("utf-8"),
        file_name=default_filename,
        mime="application/json",
        use_container_width=True
    )

with col_stub:
    if st.button("🧪 Simulate Add-to-Cart API", use_container_width=True):
        # 👉 Replace this stub with your internal JioMart API integration
        st.success("Stubbed: cart posted to /api/cart/add (simulate).")
        st.caption("Integrate here with your internal JioMart cart API using requests.")


# Footer
st.divider()
st.caption("Built for internal demo • Gemma-3 via ChatOllama • Voice + regional language support (Vosk/Google SR).")
