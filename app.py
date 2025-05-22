```python
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
import time

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
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #272727; border-radius: 4px 4px 0px 0px;
        padding: 10px 16px; font-weight: 500; color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff0000; color: white;
    }
    .stButton>button {
        background-color: #ff0000; color: white; border: none;
        border-radius: 2px; padding: 8px 16px; font-weight: 500;
    }
    .stTextInput>div>div>input {
        border-radius: 20px; background-color: #121212;
        color: #f1f1f1; border: 1px solid #303030;
    }
    .stTextArea>div>div>textarea { background-color: #121212; color: #f1f1f1; }
    .thumbnail-container {
        border: 1px solid #303030; border-radius: 8px;
        padding: 10px; background-color: #181818;
    }
    .stExpander { background-color: #181818; border: 1px solid #303030; }
    .stAlert { background-color: #181818; color: #f1f1f1; }
    .stMarkdown { color: #f1f1f1; }
    .stRadio label { color: #f1f1f1 !important; }
    .stSpinner > div { border-top-color: #f1f1f1 !important; }
    pre { background-color: #121212 !important; }
    code { color: #a9dc76 !important; }
    .generated-image-container {
        border: 2px solid #ff0000; border-radius: 8px;
        padding: 10px; background-color: #181818; margin-top: 20px;
    }
    .comparison-container { display: flex; flex-direction: row; gap: 20px; }
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
        else:
            path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if path and os.path.exists(path):
                vision_client = vision.ImageAnnotatorClient()
            else:
                st.info("Google Vision API credentials not found. Analysis will use only OpenAI.")
    except Exception as e:
        st.info(f"Google Vision API not available: {e}")
    
    # OpenAI API
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            openai.api_key = api_key
            try:
                openai_client = openai.OpenAI(api_key=api_key)
            except TypeError as e:
                if 'proxies' in str(e):
                    try:
                        openai_client = openai.OpenAI()
                    except:
                        openai_client = openai
                        st.info("Using legacy OpenAI client")
                else:
                    raise
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")
        st.info("Try updating the OpenAI library with 'pip install --upgrade openai'")
    
    return vision_client, openai_client

def extract_video_id(url):
    pattern = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    m = re.match(pattern, url)
    return m.group(6) if m else None

def get_thumbnail_url(video_id):
    urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/default.jpg"
    ]
    for u in urls:
        r = requests.head(u)
        if r.status_code == 200 and int(r.headers.get('Content-Length', 0)) > 1000:
            return u
    return None

def download_thumbnail(url):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        st.error(f"Error downloading thumbnail: {e}")
    return None

def analyze_with_vision(image_bytes, client):
    try:
        img = vision.Image(content=image_bytes)
        labels = client.label_detection(image=img).label_annotations
        text = client.text_detection(image=img).text_annotations[:1]
        faces = client.face_detection(image=img).face_annotations
        logos = client.logo_detection(image=img).logo_annotations
        props = client.image_properties(image=img).image_properties_annotation
        return {
            "labels": [{"description": l.description, "score": float(l.score)} for l in labels],
            "text": [{"description": t.description, "confidence": float(getattr(t, 'confidence', 0))} for t in text],
            "faces": [
                {
                    "joy": f.joy_likelihood.name,
                    "sorrow": f.sorrow_likelihood.name,
                    "anger": f.anger_likelihood.name,
                    "surprise": f.surprise_likelihood.name
                } for f in faces
            ],
            "logos": [{"description": lg.description} for lg in logos],
            "colors": [
                {
                    "color": {"red": c.color.red, "green": c.color.green, "blue": c.color.blue},
                    "score": float(c.score), "pixel_fraction": float(c.pixel_fraction)
                }
                for c in props.dominant_colors.colors[:5]
            ]
        }
    except Exception as e:
        st.error(f"Error analyzing with Vision API: {e}")
        return None

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_with_openai(client, b64_image):
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this YouTube thumbnail. Describe what you see in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]}
        ]
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            resp = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=500)
            return resp.choices[0].message.content
        else:
            resp = client.ChatCompletion.create(model="gpt-4o", messages=messages, max_tokens=500)
            return resp['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"OpenAI analysis error: {e}")
        return "Unable to analyze the image."

def generate_analysis(client, vision_results, openai_desc):
    prompt = f"""
Based on the provided thumbnail analyses from Google Vision AI and your own image reading, create a structured analysis covering:
- What's happening in the thumbnail
- Category of video (e.g., gaming, tutorial, vlog)
- Theme and mood
- Colors used and their significance
- Elements and objects present
- Subject impressions (emotions, expressions)
- Text present and its purpose
- Target audience

Analysis data:
{json.dumps({'vision_analysis': vision_results, 'openai_description': openai_desc}, indent=2)}
"""
    msgs = [
        {"role": "system", "content": "You are a thumbnail analysis expert."},
        {"role": "user", "content": prompt}
    ]
    try:
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            r = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=800)
            return r.choices[0].message.content
        else:
            r = client.ChatCompletion.create(model="gpt-4o", messages=msgs, max_tokens=800)
            return r['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating analysis: {e}")
        return "Unable to generate analysis."

def generate_prompt_paragraph(client, vision_results, openai_desc):
    prompt = f"""
Based on the provided thumbnail analyses from Google Vision AI and your own image reading, create a SINGLE COHESIVE PARAGRAPH that very specifically defines the thumbnail...
Analysis data:
{json.dumps({'vision_analysis': vision_results, 'openai_description': openai_desc}, indent=2)}
"""
    msgs = [
        {"role": "system", "content": "You are a thumbnail description expert."},
        {"role": "user", "content": prompt}
    ]
    try:
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            res = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=800)
            return res.choices[0].message.content
        else:
            res = client.ChatCompletion.create(model="gpt-4o", messages=msgs, max_tokens=800)
            return res['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating prompt paragraph: {e}")
        return "Unable to generate prompt paragraph."

def generate_image_from_prompt(client, prompt, image_count=1):
    """
    Generate a YouTube-style thumbnail using GPT-4o's image capability.
    """
    enhanced = (
        f"Create a hyper-realistic YouTube thumbnail with a 16:9 aspect ratio based on this description: {prompt}\n\n"
        "The image should be extremely high quality, photorealistic, and follow YouTube thumbnail best practices "
        "with vibrant colors and clear focal points. Make it look professional and eye-catching."
    )
    try:
        if hasattr(client, 'images') and hasattr(client.images, 'generate'):
            response = client.images.generate(
                model="gpt-4o",          # ← now using GPT-4o for image generation
                prompt=enhanced,
                n=image_count,
                size="1792x1024",        # 16:9 aspect ratio
                quality="hd",
                style="vivid"
            )
            return [d.url for d in response.data]
    except Exception as e:
        st.error(f"Image generation error: {e}")
        st.info("Ensure your API key has GPT-4o image permissions.")
    return None

def download_image(url):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except Exception as e:
        st.error(f"Error downloading image: {e}")
    return None

def generate_prompt_variations(client, original_prompt):
    variation_prompt = f"""
Below is a detailed description of a YouTube thumbnail:
{original_prompt}

Create TWO alternative prompt variations that could describe a thumbnail with the same core message and purpose...
"""
    msgs = [
        {"role": "system", "content": "You are a creative thumbnail designer."},
        {"role": "user", "content": variation_prompt}
    ]
    try:
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            r = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=1200)
            return r.choices[0].message.content
        else:
            r = client.ChatCompletion.create(model="gpt-4o", messages=msgs, max_tokens=1200)
            return r['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating prompt variations: {e}")
        return "VARIATION 1: Unable to generate variations.\n\nVARIATION 2: Unable to generate variations."

def main():
    st.markdown(
        '<div style="display:flex;align-items:center;padding:10px 0;">'
        '<span style="color:#FF0000;font-size:28px;font-weight:bold;margin-right:5px;">▶️</span>'
        '<h1 style="margin:0;color:#f1f1f1;">YouTube Thumbnail Analyzer & Generator</h1>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<p style="color:#aaaaaa;margin-top:0;">Analyze thumbnails and generate new ones using AI</p>',
                unsafe_allow_html=True)

    vision_client, openai_client = setup_credentials()
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return

    input_option = st.radio("Select input method:", ["Upload Image", "YouTube URL"], horizontal=True)
    image_bytes = None
    image = None
    video_info = {}

    if input_option == "Upload Image":
        uploaded = st.file_uploader("Choose a thumbnail image...", type=["jpg","jpeg","png"])
        if uploaded:
            image = Image.open(uploaded)
            buf = io.BytesIO(); image.save(buf, format=image.format or 'JPEG')
            image_bytes = buf.getvalue()
    else:
        url = st.text_input("Enter YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
        if url:
            vid = extract_video_id(url)
            if vid:
                thumb_url = get_thumbnail_url(vid)
                if thumb_url:
                    image_bytes = download_thumbnail(thumb_url)
                    if image_bytes:
                        image = Image.open(io.BytesIO(image_bytes))
                        video_info = {"id": vid, "url": url}
                else:
                    st.error("Could not retrieve thumbnail for this video.")
            else:
                st.error("Invalid YouTube URL.")

    if image_bytes and image:
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
            st.image(image, caption="Original Thumbnail", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if video_info.get("url"):
                st.markdown(
                    f'<a href="{video_info["url"]}" target="_blank" style="color:#3ea6ff;'
                    'text-decoration:none;font-weight:500;">View Original Video</a>',
                    unsafe_allow_html=True
                )

        with st.spinner("Analyzing thumbnail..."):
            b64 = encode_image(image_bytes)
            openai_desc = analyze_with_openai(openai_client, b64)
            vision_res = analyze_with_vision(image_bytes, vision_client) if vision_client else None

            with col2:
                st.subheader("Thumbnail Analysis")
                analysis = generate_analysis(openai_client, vision_res or {}, openai_desc)
                st.markdown(analysis)
                if vision_res:
                    with st.expander("View Raw Vision API Results"):
                        st.json(vision_res)
                with st.expander("View Raw OpenAI Description"):
                    st.write(openai_desc)

        with st.spinner("Generating prompt and creating new thumbnail..."):
            prompt_paragraph = generate_prompt_paragraph(openai_client, vision_res or {}, openai_desc)
            st.subheader("Generated Prompt")
            st.text_area("Thumbnail description:", value=prompt_paragraph, height=150, key="prompt_display")

            image_urls = generate_image_from_prompt(openai_client, prompt_paragraph)
            if image_urls:
                st.subheader("Generated Thumbnail")
                st.markdown('<div class="generated-image-container">', unsafe_allow_html=True)
                gen_img = download_image(image_urls[0])
                if gen_img:
                    st.image(gen_img, caption="AI-Generated Thumbnail", use_column_width=True)
                    buf = io.BytesIO(); gen_img.save(buf, format='PNG'); buf.seek(0)
                    st.download_button("Download Generated Thumbnail", data=buf,
                                       file_name="generated_thumbnail.png", mime="image/png")
                st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("Comparison")
                a, b = st.columns(2)
                with a:
                    st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                    st.image(image, caption="Original Thumbnail", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with b:
                    st.markdown('<div class="generated-image-container">', unsafe_allow_html=True)
                    st.image(gen_img, caption="AI-Generated Thumbnail", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Failed to generate image. Check GPT-4o image permissions.")

        st.subheader("Alternative Prompts")
        with st.spinner("Generating prompt variations..."):
            variations = generate_prompt_variations(openai_client, prompt_paragraph)
            parts = variations.split("VARIATION")
            if len(parts) >= 3:
                var1 = parts[1].replace("1:", "").strip()
                var2 = parts[2].replace("2:", "").strip()
                tabs = st.tabs(["Variation 1", "Variation 2"])
                with tabs[0]:
                    st.text_area("Alternative prompt 1:", value=var1, height=150, key="variation1")
                    if st.button("Generate from Variation 1"):
                        with st.spinner("Generating from variation 1..."):
                            urls = generate_image_from_prompt(openai_client, var1)
                            if urls:
                                img = download_image(urls[0])
                                if img:
                                    st.image(img, caption="Generated from Variation 1", use_column_width=True)
                                    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
                                    st.download_button("Download Thumbnail", data=buf,
                                                       file_name="variation1_thumbnail.png", mime="image/png")
                with tabs[1]:
                    st.text_area("Alternative prompt 2:", value=var2, height=150, key="variation2")
                    if st.button("Generate from Variation 2"):
                        with st.spinner("Generating from variation 2..."):
                            urls = generate_image_from_prompt(openai_client, var2)
                            if urls:
                                img = download_image(urls[0])
                                if img:
                                    st.image(img, caption="Generated from Variation 2", use_column_width=True)
                                    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
                                    st.download_button("Download Thumbnail", data=buf,
                                                       file_name="variation2_thumbnail.png", mime="image/png")
            else:
                st.warning("Unable to properly parse variations. Please try again.")

if __name__ == "__main__":
    main()
```
