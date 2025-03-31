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
from openai import OpenAI
import base64

# Set page configuration
st.set_page_config(
    page_title="YouTube Thumbnail Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for dark mode (YouTube-like)
st.markdown("""
<style>
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    .stApp { background-color: #0f0f0f; }
    h1, h2, h3 { color: #f1f1f1; font-family: 'Roboto', sans-serif; }
    p, li, div { color: #aaaaaa; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #272727;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 500;
        color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] { background-color: #ff0000; color: white; }
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
    .stExpander { background-color: #181818; border: 1px solid #303030; }
    .stAlert { background-color: #181818; color: #f1f1f1; }
    .stMarkdown { color: #f1f1f1; }
    .stRadio label { color: #f1f1f1 !important; }
    .stSpinner > div { border-top-color: #f1f1f1 !important; }
    pre { background-color: #121212 !important; }
    code { color: #a9dc76 !important; }
</style>
""", unsafe_allow_html=True)

# Setup API credentials.
def setup_credentials():
    vision_client = None
    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(credentials_dict, str):
                credentials_dict = json.loads(credentials_dict)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        elif os.path.exists("service-account.json"):
            credentials = service_account.Credentials.from_service_account_file("service-account.json")
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and os.path.exists(credentials_path):
                vision_client = vision.ImageAnnotatorClient()
            else:
                st.info("Google Vision API credentials not found. Analysis will use only OpenAI.")
    except Exception as e:
        st.info(f"Google Vision API not available: {e}")

    api_key = None
    try:
        if 'OPENAI_API_KEY' in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                api_key = st.text_input("Enter your OpenAI API key:", type="password")
                if not api_key:
                    st.warning("Please enter an OpenAI API key to continue.")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai_client = OpenAI() # Initialize OpenAI client here
            return vision_client, openai_client
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")
        return vision_client, None

    return vision_client, None # Return None if OpenAI initialization fails

# Extract YouTube video ID from URL.
def extract_video_id(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    youtube_match = re.match(youtube_regex, url)
    if youtube_match:
        return youtube_match.group(6)
    return None

# Get thumbnail URL from video ID.
def get_thumbnail_url(video_id):
    thumbnail_urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/default.jpg"
    ]
    for url in thumbnail_urls:
        response = requests.head(url)
        if response.status_code == 200 and int(response.headers.get('Content-Length', 0)) > 1000:
            return url
    return None

# Download thumbnail from URL.
def download_thumbnail(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        st.error(f"Error downloading thumbnail: {e}")
        return None

# Analyze image using Google Vision API.
def analyze_with_vision(image_bytes, vision_client):
    try:
        image = vision.Image(content=image_bytes)
        label_detection = vision_client.label_detection(image=image)
        text_detection = vision_client.text_detection(image=image)
        face_detection = vision_client.face_detection(image=image)
        logo_detection = vision_client.logo_detection(image=image)
        image_properties = vision_client.image_properties(image=image)
        results = {
            "labels": [{"description": label.description, "score": float(label.score)}
                       for label in label_detection.label_annotations],
            "text": [{"description": text.description, "confidence": float(text.confidence) if hasattr(text, 'confidence') else None}
                     for text in text_detection.text_annotations[:1]],
            "faces": [{"joy": face.joy_likelihood.name,
                       "sorrow": face.sorrow_likelihood.name,
                       "anger": face.anger_likelihood.name,
                       "surprise": face.surprise_likelihood.name}
                      for face in face_detection.face_annotations],
            "logos": [{"description": logo.description} for logo in logo_detection.logo_annotations],
            "colors": [{"color": {"red": color.color.red,
                                  "green": color.color.green,
                                  "blue": color.color.blue},
                        "score": float(color.score),
                        "pixel_fraction": float(color.pixel_fraction)}
                       for color in image_properties.image_properties_annotation.dominant_colors.colors[:5]]
        }
        return results
    except Exception as e:
        st.error(f"Error analyzing image with Google Vision API: {e}")
        return None

# Encode image to base64.
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Get a textual description of the thumbnail using OpenAI.
def analyze_with_openai(base64_image, openai_client):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Analyze this YouTube thumbnail. Describe what you see in detail. [Image: data:image/jpeg;base64,{base64_image}]"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image with OpenAI: {e}")
        return None

# Generate a photorealistic image using the latest ChatGPT-4o image generation model.
def generate_image_from_analysis(vision_results, openai_description, openai_client):
    try:
        input_data = {
            "vision_analysis": vision_results,
            "openai_description": openai_description
        }
        analysis_json = json.dumps(input_data, indent=2)
        prompt = (
            "Based on the provided analysis JSON data, generate a photorealistic digital image of a YouTube thumbnail that meets the following criteria:\n"
            "- The image must have a 16:9 aspect ratio.\n"
            "- It should be highly detailed, photorealistic, and contextually accurate.\n"
            "- Visual elements, colors, textures, lighting, and any text elements must be rendered precisely as described in the analysis.\n"
            "- The output must be returned as a base64-encoded JPEG image.\n\n"
            "Analysis data:\n" + analysis_json
        )
        # Use ChatGPT-4o for image generation.
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a state-of-the-art image generator capable of producing photorealistic images with precision and contextually accurate details."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        generated_image_base64 = response.choices[0].message.content.strip()
        return generated_image_base64
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Generate structured analysis using OpenAI.
def generate_analysis(vision_results, openai_description, openai_client):
    try:
        input_data = {
            "vision_analysis": vision_results,
            "openai_description": openai_description
        }
        prompt = (
            "Based on the provided thumbnail analyses from Google Vision AI and the image description from OpenAI, create a structured analysis covering:\n"
            "- What is happening in the thumbnail\n"
            "- Category of video (e.g., gaming, tutorial, vlog)\n"
            "- Theme and mood\n"
            "- Colors used and their significance\n"
            "- Elements and objects present\n"
            "- Subject impressions (emotions, expressions)\n"
            "- Text present and its purpose\n"
            "- Target audience\n\n"
            "Format your response with clear headings and bullet points for easy readability.\n\n"
            "Analysis data:\n" + json.dumps(input_data, indent=2)
        )
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a thumbnail analysis expert who creates detailed analyses based on image data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating analysis: {e}")
        return None

# Main application.
def main():
    st.markdown(
        '<div style="display: flex; align-items: center; padding: 10px 0;">'
        '<span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span> '
        '<h1 style="margin: 0; color: #f1f1f1;">YouTube Thumbnail Analyzer</h1>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze thumbnails using AI to understand what makes them effective</p>', unsafe_allow_html=True)

    vision_client, openai_client = setup_credentials()

    if not openai_client:
        st.error("OpenAI API key not initialized or invalid. Please check your API key.")
        return

    input_option = st.radio("Select input method:", ["Upload Image", "YouTube URL"], horizontal=True)

    image_bytes = None
    image = None
    video_info = {}

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose a thumbnail image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
            image_bytes = img_byte_arr.getvalue()
    else:
        youtube_url = st.text_input("Enter YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                video_info["id"] = video_id
                video_info["url"] = youtube_url
                thumbnail_url = get_thumbnail_url(video_id)
                if thumbnail_url:
                    video_info["thumbnail_url"] = thumbnail_url
                    image_bytes = download_thumbnail(thumbnail_url)
                    if image_bytes:
                        image = Image.open(io.BytesIO(image_bytes))
                        video_info["title"] = f"Thumbnail for Video ID: {video_id}"
                else:
                    st.error("Could not retrieve thumbnail for this video.")
            else:
                st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")

    if image_bytes and image:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
            st.image(image, caption="Thumbnail" if input_option == "Upload Image" else video_info.get("title", "YouTube Thumbnail"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if input_option == "YouTube URL" and "id" in video_info:
                st.markdown(f'<a href="{video_info["url"]}" target="_blank" style="color: #3ea6ff; text-decoration: none; font-weight: 500;">View Original Video</a>', unsafe_allow_html=True)

        with st.spinner("Analyzing thumbnail..."):
            base64_image = encode_image(image_bytes)
            openai_description = analyze_with_openai(base64_image, openai_client)
            vision_results = None
            if vision_client:
                vision_results = analyze_with_vision(image_bytes, vision_client)

            with col2:
                st.subheader("Thumbnail Analysis")
                analysis = generate_analysis(vision_results if vision_results else {"no_vision_api": True}, openai_description, openai_client)
                st.markdown(analysis)
                if vision_results:
                    with st.expander("View Raw Vision API Results"):
                        st.json(vision_results)
                with st.expander("View Raw OpenAI Description"):
                    st.write(openai_description)

        st.subheader("Generated Thumbnail")
        with st.spinner("Generating photorealistic image from analysis..."):
            generated_image_base64 = generate_image_from_analysis(
                vision_results if vision_results else {"no_vision_api": True},
                openai_description,
                openai_client
            )
            if generated_image_base64:
                generated_image_bytes = base64.b64decode(generated_image_base64)
                generated_image = Image.open(io.BytesIO(generated_image_bytes))
                st.image(generated_image, caption="Generated Photorealistic Thumbnail", use_container_width=True)

if __name__ == "__main__":
    main()
