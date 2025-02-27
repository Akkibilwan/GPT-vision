import streamlit as st
import os
import io
import json
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Function to setup API credentials
def setup_credentials():
    vision_client = None
    openai_client = None
    
    # For Google Vision API
    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            # Use the provided secrets
            credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(credentials_dict, str):
                credentials_dict = json.loads(credentials_dict)
            
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            st.success("Google Vision API credentials loaded successfully.")
        else:
            # Check for local file
            if os.path.exists("service-account.json"):
                credentials = service_account.Credentials.from_service_account_file("service-account.json")
                vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                st.success("Google Vision API credentials loaded from local file.")
            else:
                # Look for credentials in environment variable
                credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                if credentials_path and os.path.exists(credentials_path):
                    vision_client = vision.ImageAnnotatorClient()
                    st.success("Google Vision API credentials loaded from environment variable.")
                else:
                    st.error("Google Vision API credentials not found.")
    except Exception as e:
        st.error(f"Error loading Google Vision API credentials: {e}")
    
    # For OpenAI API
    try:
        api_key = None
        if 'OPENAI_API_KEY' in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.success("OpenAI API key loaded successfully.")
        else:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                st.success("OpenAI API key loaded from environment variable.")
            else:
                api_key = st.text_input("Enter your OpenAI API key:", type="password")
                if not api_key:
                    st.warning("Please enter an OpenAI API key to continue.")
        
        if api_key:
            openai_client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")
    
    return vision_client, openai_client

# Function to analyze image with Google Vision API
def analyze_with_vision(image_bytes, vision_client):
    try:
        image = vision.Image(content=image_bytes)
        
        # Perform different types of detection
        label_detection = vision_client.label_detection(image=image)
        text_detection = vision_client.text_detection(image=image)
        face_detection = vision_client.face_detection(image=image)
        logo_detection = vision_client.logo_detection(image=image)
        image_properties = vision_client.image_properties(image=image)
        
        # Extract results
        results = {
            "labels": [{"description": label.description, "score": float(label.score)} 
                      for label in label_detection.label_annotations],
            "text": [{"description": text.description, "confidence": float(text.confidence) if hasattr(text, 'confidence') else None}
                    for text in text_detection.text_annotations[:1]],  # Just get the full text
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

# Function to encode image to base64 for OpenAI
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to analyze image with OpenAI
def analyze_with_openai(client, base64_image):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this YouTube thumbnail. Describe what you see in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image with OpenAI: {e}")
        return None

# Function to generate detailed prompt using OpenAI based on both analyses
def generate_prompt(client, vision_results, openai_description):
    try:
        # Prepare input for GPT
        input_data = {
            "vision_analysis": vision_results,
            "openai_description": openai_description
        }
        
        prompt = """
        Based on the provided thumbnail analyses from Google Vision AI and your own image reading, create TWO distinct outputs:
        
        1. ANALYSIS - A structured analysis covering:
           - What's happening in the thumbnail
           - Category of video (e.g., gaming, tutorial, vlog) 
           - Theme and mood
           - Colors used and their significance
           - Elements and objects present
           - Subject impressions (emotions, expressions)
           - Text present and its purpose
           - Target audience
        
        2. PROMPT - A cohesive, detailed paragraph (not bullet points) that very specifically defines the thumbnail, describing:
           - The exact theme
           - Specific colors used and how they interact
           - All visual elements and their arrangement
           - Overall style and artistic approach
           - Text elements and their presentation
           - Emotional impact the thumbnail is designed to create
        
        Format your response with clear separation between the ANALYSIS and PROMPT sections.
        The PROMPT paragraph should be comprehensive enough that someone could recreate the thumbnail from the description alone.
        
        Analysis data:
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a thumbnail analysis expert who can create detailed prompts based on image analysis data."},
                {"role": "user", "content": prompt + json.dumps(input_data, indent=2)}
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating prompt: {e}")
        return None

# Main app
def main():
    st.title("YouTube Thumbnail Analyzer")
    st.write("Upload a thumbnail to analyze it using Google Vision AI and OpenAI, and generate a detailed prompt.")
    
    # Initialize and check API clients
    vision_client, openai_client = setup_credentials()
    
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a thumbnail image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Thumbnail", use_column_width=True)
        
        # Convert to bytes for API processing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        with st.spinner("Analyzing thumbnail..."):
            # Process with OpenAI (always available if we've gotten this far)
            base64_image = encode_image(img_byte_arr)
            openai_description = analyze_with_openai(openai_client, base64_image)
            
            # Process with Google Vision API (if available)
            vision_results = None
            if vision_client:
                vision_results = analyze_with_vision(img_byte_arr, vision_client)
            
            # Show raw analysis results in expanders
            with col2:
                if vision_results:
                    with st.expander("Google Vision API Results"):
                        st.json(vision_results)
                
                with st.expander("OpenAI Description"):
                    st.write(openai_description)
            
            # Generate the detailed prompt
            with st.spinner("Analyzing thumbnail and generating results..."):
                time.sleep(1)  # Small delay for better UX
                
                if vision_results:
                    # Use both analyses if Vision API is available
                    combined_analysis = generate_prompt(openai_client, vision_results, openai_description)
                else:
                    # Use only OpenAI description if Vision API is not available
                    combined_analysis = generate_prompt(openai_client, {"no_vision_api": True}, openai_description)
                
                if combined_analysis:
                    try:
                        # Split the response into Analysis and Prompt sections
                        sections = combined_analysis.split("PROMPT")
                        
                        if len(sections) > 1:
                            analysis_part = sections[0].replace("ANALYSIS", "").strip()
                            prompt_part = sections[1].strip()
                            
                            # Display the Analysis section
                            st.subheader("Detailed Analysis")
                            st.markdown(analysis_part)
                            
                            # Display the Prompt section
                            st.subheader("Thumbnail Prompt")
                            
                            # Display prompt in a text area for easy copying
                            st.text_area("Copy this prompt:", value=prompt_part, height=200)
                            
                            # Add a download button for just the prompt
                            st.download_button(
                                label="Download Prompt",
                                data=prompt_part,
                                file_name="thumbnail_prompt.txt",
                                mime="text/plain"
                            )
                        else:
                            # If splitting didn't work as expected, show the entire output
                            st.markdown(combined_analysis)
                    except Exception as e:
                        st.error(f"Error parsing the analysis output: {e}")
                        st.markdown(combined_analysis)

if __name__ == "__main__":
    main()
