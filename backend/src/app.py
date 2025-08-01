import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import speech_recognition as sr
import threading
import wave
import numpy as np
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please set GOOGLE_API_KEY environment variable")

print(f"Initializing Gemini with API key:")
genai.configure(api_key=api_key)

app = Flask(__name__)
CORS(app)

# Configure Gemini api
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
chat = model.start_chat(history=[])

# Store context globally to access a number of methods

context = {
    "resume": "",
    "job_description": ""
}

class InterviewAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        
    def start_listening(self):
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            return str(e)

    def generate_introduction(self):
        if not context["resume"] or not context["job_description"]:
            return "Please upload resume and job description first."
        
        prompt = f"""
        Based on this resume:
        {context['resume']}
        And this job description:
        {context['job_description']}
        Generate a professional 1-minute introduction for a job interview."""
        
        response = chat.send_message(prompt, stream=True)
        return ' '.join([chunk.text for chunk in response])

assistant = InterviewAssistant()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_context', methods=['POST'])
def upload_context():
    data = request.json
    context["resume"] = data.get('resume', '')
    context["job_description"] = data.get('jobDescription', '')
    return jsonify({"status": "success", "message": "Context uploaded successfully"})

@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        question = request.json['question']
        
        if not context["resume"] or not context["job_description"]:
            return jsonify({"error": "Please upload resume and job description first"}), 400
        
        prompt = f"""
        Based on this resume:
        {context['resume']} 
        And this job description:
        {context['job_description']}  
        Please provide a professional response to this interview question and ensure that it is short and relevant and sounds human:
        {question}
        """
        
        response = chat.send_message(prompt, stream=True)
        response_text = ' '.join([chunk.text for chunk in response])
        context["response"] = response_text
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/generate_followup', methods=['POST'])
def generate_followup():
    try:
        question = request.json['question']
        
        if not context["resume"] or not context["job_description"]:
            return jsonify({"error": "Please upload resume and job description first"}), 400
        if not context.get("response"):
            return jsonify({"error": "No response available for follow-up."}), 400 
        prompt = f"""
        Based on this response:
        {context['response']}
        
        Please provide a professional followup response to this interview question giving weightage to the question and the context on which it is asked ensure that it is short and sounds human:
        {question}
        """
        
        response1 = chat.send_message(prompt, stream=True)
        response_text1 = ' '.join([chunk.text for chunk in response1])
        
        return jsonify({"response": response_text1})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/start_listening', methods=['POST'])
def start_listening():
    text = assistant.start_listening()
    return jsonify({"text": text})

@app.route('/generate_introduction', methods=['POST'])
@app.route('/generate_introduction', methods=['POST'])
def get_introduction():
    try:
        if not context["resume"] or not context["job_description"]:
            return jsonify({"error": "Please upload resume and job description first"}), 400
        
        prompt = f"""
        Based on this resume:
        {context['resume']}
        
        And this job description:
        {context['job_description']}
        
        Generate a professional 1-minute introduction for a job interview. 
        The introduction should highlight relevant experience and skills that match the job requirements.
        """
        
        response = chat.send_message(prompt, stream=True)
        introduction_text = ' '.join([chunk.text for chunk in response])
        
        return jsonify({"introduction": introduction_text})
    except Exception as e:
        print(f"Error generating introduction: {str(e)}")  # For debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)