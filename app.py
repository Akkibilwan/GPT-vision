import streamlit as st
import os
import io
import json
import re
import requests
import time
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
from openai.error import APIError, RateLimitError, OpenAIError
import base64

# --- PAGE CONFIG & CSS -----------------------------------------------------------

st.set_page_config(
    page_title="YouTube Thumbnail Analyzer & Generator",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    /* Dark-mode YouTube-style */
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    .stApp { background-color: #0f0f0f; }
    h1,h2,h3 { color: #f1f1f1; font-family: 'Roboto', sans-serif; }
    p,li,div { color: #aaaaaa; }
    .stTabs [data-baseweb="tab"] { background-color: #272727; color: #f1f1f1; border-radius:4px 4px 0 0; padding:8px 16px; }
    .stTabs [aria-selected="true"] { background-color: #ff0000; color:white; }
    .stButton>button { background-color:#ff0000; color:white; border:none; border-radius:2px; padding:8px 16px; }
    .stTextInput input, .stTextArea textarea { background-color:#121212; color:#f1f1f1; border:1px solid #303030; border-radius:20px; }
    .thumbnail-container, .generated-image-container, .stExpander, .stAlert { background-color:#181818; border:1px solid #303030; border-radius:8px; padding:10px; }
    .stRadio label { color:#f1f1f1 !important; }
    pre { background-color:#121212 !important; }
    code { color:#a9dc76 !important; }
</style>
""", unsafe_allow_html=True)

# --- HELPER: OPENAI RETRY --------------------------------------------------------

def call_with_retries(func, *args, max_retries=3, backoff=1, **kwargs):
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            status = getattr(e, "http_status", 500)
            if 500 <= status < 600:
                time.sleep(backoff * (2**i))
                continue
            raise
        except RateLimitError:
            time.sleep(backoff * (2**i))
            continue
    raise OpenAIError("Retries exhausted")

# --- CREDENTIALS SETUP -----------------------------------------------------------

def setup_credentials():
    vision_client = None
    # Google Vision
    try:
        if "GOOGLE_CREDENTIALS" in st.secrets:
            creds = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(creds, str):
                creds = json.loads(creds)
            creds = service_account.Credentials.from_service_account_info(creds)
            vision_client = vision.ImageAnnotatorClient(credentials=creds)
        elif os.path.exists("service-account.json"):
            creds = service_account.Credentials.from_service_account_file("service-account.json")
            vision_client = vision.ImageAnnotatorClient(credentials=creds)
        else:
            st.info("Google Vision credentials not found; continuing without Vision API.")
    except Exception as e:
        st.warning(f"Vision API init failed: {e}")

    # OpenAI
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        openai.api_key = api_key
    else:
        st.error("OpenAI API key required.")
        st.stop()

    return vision_client

# --- YOUTUBE THUMBNAIL HELPERS --------------------------------------------------

def extract_video_id(url):
    m = re.match(
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([^&]{11})',
        url
    )
    return m.group(1) if m else None

def get_thumbnail_url(vid):
    for size in ("maxresdefault","hqdefault","mqdefault","default"):
        url = f"https://img.youtube.com/vi/{vid}/{size}.jpg"
        r = requests.head(url, allow_redirects=True)
        if r.status_code == 200 and int(r.headers.get("Content-Length","0")) > 1000:
            return url
    return None

def download_bytes(url):
    r = requests.get(url, stream=True)
    return r.content if r.status_code == 200 else None

def encode_image(bts):
    return base64.b64encode(bts).decode()

# --- ANALYSIS FUNCTIONS ---------------------------------------------------------

def analyze_with_openai(base64_image: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this YouTube thumbnail in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} 
        ]
    }]
    try:
        resp = call_with_retries(
            openai.ChatCompletion.create,
            model="gpt-4o", messages=messages, max_tokens=500
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        st.error(f"OpenAI error: {e}")
        return ""

def analyze_with_vision(image_bytes, client) -> dict:
    img = vision.Image(content=image_bytes)
    labels = client.label_detection(image=img).label_annotations[:3]
    colors = client.image_properties(image=img)\
                   .image_properties_annotation.dominant_colors.colors[:3]
    return {
        "labels": [{"desc": l.description, "score": l.score} for l in labels],
        "colors": [
            {"rgb": (c.color.red, c.color.green, c.color.blue), "score": c.score}
            for c in colors
        ]
    }

def generate_analysis(vision_res: dict, text_res: str) -> str:
    payload = {
        "labels": vision_res.get("labels", []),
        "colors": vision_res.get("colors", []),
        "description": text_res
    }
    prompt = (
        "You are a thumbnail analysis expert. Given this data, "
        "produce a structured analysis with headings and bullets:\n"
        + json.dumps(payload)
    )
    messages = [
        {"role": "system", "content": "You are a thumbnail analysis expert."},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = call_with_retries(
            openai.ChatCompletion.create,
            model="gpt-4o", messages=messages, max_tokens=600
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        st.error(f"Analysis generation failed: {e}")
        return "Analysis unavailable."

# --- PROMPT PARAGRAPH ------------------------------------------------------------

def generate_prompt_paragraph(vision_res: dict, text_res: str) -> str:
    payload = {
        "labels": vision_res.get("labels", []),
        "colors": vision_res.get("colors", []),
        "description": text_res
    }
    prompt = (
        "You are a thumbnail description expert. "
        "Write one cohesive paragraph so someone could recreate the thumbnail:\n"
        + json.dumps(payload)
    )
    messages = [
        {"role": "system", "content": "You write precise thumbnail descriptions."},
        {"role": "user",   "content": prompt}
    ]
    try:
        resp = call_with_retries(
            openai.ChatCompletion.create,
            model="gpt-4o", messages=messages, max_tokens=600
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        st.error(f"Description generation failed: {e}")
        return ""

# --- IMAGE GENERATION ------------------------------------------------------------

def generate_image_from_prompt(prompt: str, n=1) -> list[str]:
    enhanced = (
        f"Hyper-realistic YouTube thumbnail with 16:9 aspect ratio HD. {prompt}"
    )
    try:
        resp = call_with_retries(
            openai.Image.create,
            prompt=enhanced, n=n, size="1792x1024", response_format="url"
        )
        return [d["url"] for d in resp["data"]]
    except OpenAIError as e:
        st.error(f"Image generation failed: {e}")
        return []

# --- VARIATIONS ------------------------------------------------------------------

def generate_prompt_variations(original: str) -> str:
    variation_prompt = (
        "Generate TWO distinct alternative thumbnail descriptions "
        "based on this one:\n" + original
    )
    messages = [
        {"role": "system", "content": "You craft varied thumbnail prompts."},
        {"role": "user",   "content": variation_prompt}
    ]
    try:
        resp = call_with_retries(
            openai.ChatCompletion.create,
            model="gpt-4o", messages=messages, max_tokens=800
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        st.error(f"Variation generation failed: {e}")
        return "VARIATION 1: …\n\nVARIATION 2: …"

# --- MAIN APP -------------------------------------------------------------------

def main():
    st.markdown(
        '<div style="display:flex;align-items:center">'
        '<span style="font-size:28px;color:#FF0000">▶️</span>'
        '<h1 style="color:#f1f1f1;margin-left:8px">'
        'YouTube Thumbnail Analyzer & Generator</h1></div>',
        unsafe_allow_html=True
    )

    vision_client = setup_credentials()

    choice = st.radio("Select Input Method:", ["Upload Image", "YouTube URL"], horizontal=True)
    image_bytes = None

    if choice == "Upload Image":
        up = st.file_uploader("Upload a thumbnail image...", type=["jpg", "jpeg", "png"])
        if up:
            image_bytes = up.read()
    else:
        url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
        if vid := extract_video_id(url):
            thumb_url = get_thumbnail_url(vid)
            if thumb_url:
                image_bytes = download_bytes(thumb_url)
            else:
                st.error("Could not retrieve thumbnail for this video.")
        elif url:
            st.error("Invalid YouTube URL.")

    if not image_bytes:
        return

    img = Image.open(io.BytesIO(image_bytes))
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, caption="Original Thumbnail", use_column_width=True)

    # --- ANALYZE ---------------------------------------------------------------

    with st.spinner("Analyzing thumbnail..."):
        b64 = encode_image(image_bytes)
        text_desc = analyze_with_openai(b64)
        vision_res = analyze_with_vision(image_bytes, vision_client) if vision_client else {"labels": [], "colors": []}
        structured = generate_analysis(vision_res, text_desc)

    with col2:
        st.subheader("Structured Analysis")
        st.markdown(structured)

    # --- PROMPT & GENERATE ---------------------------------------------------

    prompt_para = generate_prompt_paragraph(vision_res, text_desc)
    st.subheader("Generated Prompt")
    st.text_area("Thumbnail description:", prompt_para, height=150)

    urls = generate_image_from_prompt(prompt_para)
    if urls:
        gen_bytes = download_bytes(urls[0])
        gen_img = Image.open(io.BytesIO(gen_bytes))
        st.subheader("AI-Generated Thumbnail")
        st.image(gen_img, use_column_width=True)
        buf = io.BytesIO()
        gen_img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Generated Thumbnail", buf, "generated_thumbnail.png", mime="image/png")

    # --- PROMPT VARIATIONS ---------------------------------------------------

    st.subheader("Alternative Prompts")
    vars_text = generate_prompt_variations(prompt_para)
    parts = vars_text.split("VARIATION 2:", 1)
    v1 = parts[0].replace("VARIATION 1:", "").strip()
    v2 = parts[1].strip() if len(parts) > 1 else ""

    tabs = st.tabs(["Variation 1", "Variation 2"])
    with tabs[0]:
        st.text_area("Variation 1:", v1, height=120)
    with tabs[1]:
        st.text_area("Variation 2:", v2, height=120)

if __name__ == "__main__":
    main()
