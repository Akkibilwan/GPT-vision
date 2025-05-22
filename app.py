# Updated Streamlit App with Corrected Image Size for gpt-image-1

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
    page_title="YouTube Thumbnail Analyzer & Generator",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for YouTube-like styling with dark mode
st.markdown("""
<style>
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    .stApp { background-color: #0f0f0f; }
    h1, h2, h3 { color: #f1f1f1; font-family: 'Roboto', sans-serif; }
    p, li, div { color: #aaaaaa; }
    .thumbnail-container { border: 1px solid #303030; border-radius: 8px; padding: 10px; background-color: #181818; }
    .generated-image-container { border: 2px solid #ff0000; border-radius: 8px; padding: 10px; background-color: #181818; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

def setup_credentials():
    vision_client = None
    openai_client = None
    # Google Vision API
    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            creds = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(creds, str):
                creds = json.loads(creds)
            credentials = service_account.Credentials.from_service_account_info(creds)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        elif os.path.exists("service-account.json"):
            credentials = service_account.Credentials.from_service_account_file("service-account.json")
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    except Exception:
        pass

    # OpenAI API
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            openai.api_key = api_key
            try:
                openai_client = openai.OpenAI(api_key=api_key)
            except TypeError:
                openai_client = openai
    except Exception:
        pass

    return vision_client, openai_client

def extract_video_id(url):
    m = re.match(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})', url)
    return m.group(6) if m else None

def get_thumbnail_url(video_id):
    for quality in ["maxresdefault", "hqdefault", "mqdefault", "default"]:
        url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
        r = requests.head(url)
        if r.status_code == 200 and int(r.headers.get('Content-Length', 0)) > 1000:
            return url
    return None

def download_thumbnail(url):
    r = requests.get(url, stream=True)
    return r.content if r.status_code == 200 else None

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_with_openai(client, b64_image):
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this YouTube thumbnail in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]}
        ]
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            resp = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=500)
            return resp.choices[0].message.content
        else:
            resp = client.ChatCompletion.create(model="gpt-4o", messages=messages, max_tokens=500)
            return resp['choices'][0]['message']['content']
    except Exception:
        return "Unable to analyze the image."

def generate_analysis(client, analysis_data):
    prompt = f"""
Based on the provided analyses, create a structured analysis:
- What's happening
- Category
- Theme
- Colors
- Elements
- Emotions
- Text purpose
- Audience

Data:
{json.dumps(analysis_data, indent=2)}
"""
    msgs = [
        {"role": "system", "content": "You are an expert thumbnail analyst."},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=800)
    return resp.choices[0].message.content

def generate_prompt_paragraph(client, analysis_data):
    prompt = f"""
Create a single cohesive paragraph describing the thumbnail.
Data:
{json.dumps(analysis_data, indent=2)}
"""
    msgs = [
        {"role": "system", "content": "You are a description expert."},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=800)
    return resp.choices[0].message.content

def generate_image_from_prompt(client, prompt, image_count=1):
    """
    Generate a YouTube-style thumbnail using ChatGPT’s image capability.
    """
    enhanced = (
        f"Create a hyper-realistic YouTube thumbnail with a 16:9 aspect ratio based on this description: {prompt}\n\n"
        "The image should be extremely high quality, photorealistic, and follow YouTube thumbnail best practices "
        "with vibrant colors and clear focal points. Make it look professional and eye-catching."
    )
    try:
        response = client.images.generate(
            model="gpt-image-1",    # supported ChatGPT image model
            prompt=enhanced,
            n=image_count,
            size="1536x1024"        # closest landscape size supported
        )
        return [d.url for d in response.data]
    except Exception as e:
        st.error(f"Image generation error: {e}")
        st.info("Ensure your API key has gpt-image-1 permissions and use one of: '1024x1024','1024x1536','1536x1024','auto'.")
        return None

def download_image(url):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except Exception:
        pass
    return None

def main():
    st.title("YouTube Thumbnail Analyzer & Generator")
    vision_client, openai_client = setup_credentials()
    if not openai_client:
        st.error("OpenAI client not initialized.")
        return

    input_option = st.radio("Input:", ["Upload Image", "YouTube URL"], horizontal=True)
    image_bytes = None

    if input_option == "Upload Image":
        uploaded = st.file_uploader("Upload thumbnail", type=["jpg","png","jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            buf = io.BytesIO(); img.save(buf, format='PNG')
            image_bytes = buf.getvalue()
    else:
        url = st.text_input("YouTube URL")
        vid = extract_video_id(url)
        if vid:
            thumb_url = get_thumbnail_url(vid)
            if thumb_url:
                image_bytes = download_thumbnail(thumb_url)

    if image_bytes:
        b64 = encode_image(image_bytes)
        openai_desc = analyze_with_openai(openai_client, b64)
        analysis_data = {"openai_description": openai_desc}
        st.subheader("Analysis")
        st.write(generate_analysis(openai_client, analysis_data))

        st.subheader("Prompt Paragraph")
        para = generate_prompt_paragraph(openai_client, analysis_data)
        st.write(para)

        urls = generate_image_from_prompt(openai_client, para)
        if urls:
            st.subheader("Generated Thumbnail")
            gen_img = download_image(urls[0])
            st.image(gen_img, use_column_width=True)
            buf2 = io.BytesIO(); gen_img.save(buf2, format='PNG'); buf2.seek(0)
            st.download_button("Download", data=buf2, file_name="thumbnail.png", mime="image/png")

if __name__ == "__main__":
    main()
