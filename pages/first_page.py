# MAKE THE NECESSARY PACKAGE IMPORTS
import os
import streamlit as st
from clarifai.client.model import Model
import base64
from dotenv import load_dotenv

load_dotenv()

# LOAD OPENAI AND CLARIFAI API KEYS
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# USE GPT-4 TURBO TO REFINE THE USER INPUT
def process_user_input(user_description, api_key):
    prompt = f"Given the input: {user_description}, answer the question but keep the response very short and" \
             f" maintain the context"
    inference_params = dict(temperature=0.2, max_tokens=100, api_key=api_key)
    # Model Predict
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-turbo") \
        .predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
    return model_prediction.outputs[0].data.text.raw


# USE DALL-E-3 TO GENERATE MEDICAL IMAGE BASED ON THE REFINED USER INPUT
def generate_image(processed_input, api_key):
    prompt = f"Given the input: {processed_input}, create a proper image that reflects the prompt. " \
             f"Ensure there is a person in the image. Keep the context and make the image realistic"
    inference_params = dict(quality="standard", size="1792x1024")
    # Model Predict
    model_prediction = Model(f"https://clarifai.com/openai/dall-e/models/dall-e-3?api_key={api_key}") \
        .predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    output_base64 = model_prediction.outputs[0].data.image.base64
    # Save the image as a binary file
    with open("generated_image.png", "wb") as file:
        file.write(output_base64)
    return "generated_image.png"


# USE GPT-4 VISION TO INTERPRET THE IMAGE
def understand_image(image_path, api_key):
    # Load the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = "Analyze the content of this image and give an educative description of the image given" \
             " while keeping the context."
    inference_params = dict(temperature=0.2, image_base64=base64_image, api_key=api_key)
    # Model Predict
    model_prediction = Model(f"https://clarifai.com/openai/chat-completion/models/gpt-4-vision") \
        .predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    return model_prediction.outputs[0].data.text.raw


# USE TEXT-TO-SPEECH TO MAKE AUDIO OUTPUT OF GPT-4 VISION INTERPRETATION
def text_to_speech(input_text, api_key):
    inference_params = dict(voice='alloy', speed=1.0, api_key=api_key)
    # Model Predict
    model_prediction = Model(f"https://clarifai.com/openai/tts/models/openai-tts-1") \
        .predict_by_bytes(
        input_text.encode(), input_type="text", inference_params=inference_params
    )
    audio_base64 = model_prediction.outputs[0].data.audio.base64
    return audio_base64


# STREAMLIT UI
def main():
    st.set_page_config(page_title="Interactive Medical Image Generation", layout="wide")
    st.title("Welcome To MedVision AI! 🤖🏥")

    with st.sidebar:
        st.header("Controls")
        image_description = st.text_area("WHAT IS THE MEDICAL PROBLEM?", height=100)
        generate_image_btn = st.button("Generate Image")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Medical Image")
        if generate_image_btn and image_description:
            with st.spinner("Generating image..."):
                processed_text = process_user_input(user_description=image_description, api_key=OPENAI_API_KEY)
                image_path = generate_image(processed_input=processed_text, api_key=CLARIFAI_PAT)
                if image_path:
                    st.image(
                        image_path,
                        caption="Generated Medical Image",
                        use_column_width=True,
                    )
                    st.success("Image generated!")
                else:
                    st.error("Failed to generate image. Try again later")

    with col2:
        st.header("Image Explanation")
        if generate_image_btn and image_description:
            with st.spinner("Understanding the image..."):
                understood_text = understand_image(image_path=image_path, api_key=OPENAI_API_KEY)
                audio_base64 = text_to_speech(input_text=understood_text, api_key=OPENAI_API_KEY)
                st.audio(audio_base64, format="audio/mp3")
                st.success("Audio generated from image understanding!")


if __name__ == "__main__":
    main()
