import streamlit as st
import os
import io
import json
import re
import requests
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
import base64

# Set page configuration
st.set_page_config(
    page_title="YouTube Thumbnail Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for YouTube-like styling with dark mode
st.markdown("""
<style>
    .main {
        background-color: #0f0f0f;
        color: #f1f1f1;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    h1, h2, h3 {
        color: #f1f1f1;
        font-family: 'Roboto', sans-serif;
    }
    p, li, div {
        color: #aaaaaa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #272727;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
        color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff0000;
        color: white;
    }
    .stButton>button {
        background-color: #ff0000;
        color: white;
        border: none;
        border-radius: 2px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        background-color: #121212;
        color: #f1f1f1;
        border: 1px solid #303030;
    }
    .stTextArea>div>div>textarea {
        background-color: #121212;
        color: #f1f1f1;
    }
    .thumbnail-container {
        border: 1px solid #303030;
        border-radius: 8px;
        padding: 10px;
        background-color: #181818;
    }
    .stExpander {
        background-color: #181818;
        border: 1px solid #303030;
    }
    .stAlert {
        background-color: #181818;
        color: #f1f1f1;
    }
    .stMarkdown {
        color: #f1f1f1;
    }
    .stRadio label {
        color: #f1f1f1 !important;
    }
    .stSpinner > div {
        border-top-color: #f1f1f1 !important;
    }
    pre {
        background-color: #121212 !important;
    }
    code {
        color: #a9dc76 !important;
    }
</style>
""", unsafe_allow_html=True)

# Setup credentials

def setup_credentials():
    vision_client = None
    openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            credentials_dict = json.loads(st.secrets['GOOGLE_CREDENTIALS'])
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        elif os.path.exists("service-account.json"):
            credentials = service_account.Credentials.from_service_account_file("service-account.json")
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        st.info(f"Google Vision API not available: {e}")

    return vision_client

# YouTube helpers

def extract_video_id(url):
    pattern = r'(?:v=|youtu.be/|embed/)([\w-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_thumbnail_url(video_id):
    urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    ]
    for url in urls:
        if requests.head(url).status_code == 200:
            return url
    return None

def download_thumbnail(url):
    try:
        response = requests.get(url)
        return response.content if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

# Vision API analysis

def analyze_with_vision(image_bytes, client):
    try:
        image = vision.Image(content=image_bytes)
        label_response = client.label_detection(image=image)
        text_response = client.text_detection(image=image)
        face_response = client.face_detection(image=image)
        logo_response = client.logo_detection(image=image)
        properties = client.image_properties(image=image)

        return {
            "labels": [l.description for l in label_response.label_annotations],
            "text": text_response.text_annotations[0].description if text_response.text_annotations else "",
            "faces": [f.joy_likelihood for f in face_response.face_annotations],
            "logos": [l.description for l in logo_response.logo_annotations],
            "colors": [
                {
                    "rgb": {
                        "r": c.color.red,
                        "g": c.color.green,
                        "b": c.color.blue
                    },
                    "score": c.score
                } for c in properties.image_properties_annotation.dominant_colors.colors
            ]
        }
    except Exception as e:
        st.error(f"Vision API error: {e}")
        return {}

# GPT-4o Image generation

def generate_image_from_analysis(vision_results):
    try:
        prompt = f"""
        Create a YouTube thumbnail with:
        - Aspect ratio 16:9 (1280x720 resolution)
        - Based on this analysis:
        {json.dumps(vision_results, indent=2)}
        - No text unless mentioned
        - Strong colors, emotions, high contrast
        - Style typical of high-performing YouTube thumbnails
        """

        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1280x720"
        )

        return response.data[0].url if response and response.data else None

    except Exception as e:
        st.error(f"Image generation error: {e}")
        return None

# Main app

def main():
    st.markdown('<div style="display: flex; align-items: center; padding: 10px 0;"><span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span> <h1 style="margin: 0; color: #f1f1f1;">YouTube Thumbnail Analyzer</h1></div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Upload a thumbnail or paste a YouTube URL. Get AI-generated suggestions based on Vision + GPT-4o.</p>', unsafe_allow_html=True)

    vision_client = setup_credentials()
    if not openai.api_key:
        st.warning("OpenAI API key not found.")
        return

    input_option = st.radio("Select input method:", ["Upload Image", "YouTube URL"], horizontal=True)
    image_bytes = None
    image = None

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            image_bytes = buf.getvalue()
    else:
        youtube_url = st.text_input("Enter YouTube video URL:")
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                thumbnail_url = get_thumbnail_url(video_id)
                if thumbnail_url:
                    image_bytes = download_thumbnail(thumbnail_url)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    st.error("Thumbnail not found.")
            else:
                st.error("Invalid YouTube URL.")

    if image_bytes and image:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Original Thumbnail", use_column_width=True)

        with st.spinner("Analyzing with Vision AI..."):
            vision_results = analyze_with_vision(image_bytes, vision_client)

        with st.spinner("Generating AI-enhanced thumbnail with GPT-4o..."):
            image_url = generate_image_from_analysis(vision_results)

        with col2:
            if image_url:
                st.image(image_url, caption="AI Generated Thumbnail", use_column_width=True)
                st.markdown(f"[Download Thumbnail]({image_url})", unsafe_allow_html=True)
            else:
                st.warning("Thumbnail generation failed.")

if __name__ == "__main__":
    main()
