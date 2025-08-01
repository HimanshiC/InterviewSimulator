import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables!")

# Configure the API
genai.configure(api_key=api_key)

def list_models():
    print("Available models:")
    try:
        models = genai.list_models()
        for m in models:
            print("-", m.name)
    except Exception as e:
        print(" Error listing models:", e)

def test_prompt(model_name):
    try:
        print(f"\n🔍 Trying model: {model_name}")
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat()
        response = chat.send_message("What are the benefits of using Python for AI development?")
        print(" Response:")
        print(response.text)
    except Exception as e:
        print(f" Failed with model {model_name}: {e}")

if __name__ == "__main__":
    list_models()

    # Try common models — update if needed based on list_models output
    candidate_models = [
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-1.5-pro"
    ]

    for model in candidate_models:
        test_prompt(model)
