"""import openai
import os
OPENAI_API_KEY = 'pk-jlDQmgBAgzgSbIJYtzmPPeazjrqlbZnfiuCmakbGEusEsDrm'
openai.api_key = OPENAI_API_KEY
openai.base_url = "http://localhost:3040/v1/"

completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "How do I list all files in a directory using Python?"},
    ],
)
print(os.getenv("OPENAI_API_KEY"))"""
#print(completion.choices[0].message.content)
"""import openai

openai.api_key = 'pk-jlDQmgBAgzgSbIJYtzmPPeazjrqlbZnfiuCmakbGEusEsDrm'
openai.base_url = "http://localhost:3040/v1/"

completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "How do I list all files in a directory using Python?"},
    ],
)

print(completion.choices[0].message.content)"""
import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import sounddevice as sd
import numpy as np
import json
from flask import render_template
import speech_recognition as sr
import wave
import threading
app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Function to load gemini pro model and get response
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

context = {
    "resume": "",
    "job_description": ""
}

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
        
        # Create prompt using context
        prompt = f"""
        Based on this resume:
        {context['resume']}
        
        And this job description:
        {context['job_description']}
        
        Please provide a professional response to this interview question:
        {question}
        """
        
        response = chat.send_message(prompt, stream=True)
        response_text = ' '.join([chunk.text for chunk in response])
        
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def generate_introduction():
        prompt = f"""
        Based on this resume: {assistant.resume}
        And this job description: {assistant.job_description}
    
        Generate a professional 1-minute introduction for a job interview.
        """
        return generate_response(prompt)
"""def generate_response(question):
        prompt = f"""
        #Resume: {assistant.resume}
        #Job Description: {assistant.job_description}
        #Interview Question: {question}
    
        #Please provide a professional response suitable for a job interview.
"""
        return safe_generate_content(prompt)"""
"""@app.route('/api/chat', methods=['POST'])
def chat_response():
    data = request.json
    question = data.get('question', '')
    response = chat.send_message(question, stream=True)
    return jsonify({'response': ' '.join([chunk.text for chunk in response])})
def get_gemini_response(question):
  response = chat.send_message(question, stream=True)
  return response"""
"""class InterviewAssistant:
    def __init__(self):
        self.resume = None
        self.job_description = None
        
    def set_context(self, resume, job_description):
        self.resume = resume
        self.job_description = job_description
    def safe_generate_content(prompt):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating content: {str(e)}"
 """  """def generate_response(self, question):
        prompt = f"""
        #Resume: {self.resume}
        #Job Description: {self.job_description}
        #Interview Question: {question}
        
        #Please provide a professional response suitable for a job interview.
        #"""
        #response = model.generate_content(prompt)
        #return response.text"""
    
        
    #def generate_introduction(self):
    #    prompt = f"""
    #    Based on this resume: {self.resume}
     #   And this job description: {self.job_description}
        
      #  Generate a professional 1-minute introduction for a job interview.
       # """
        #response = model.generate_content(prompt)
        #return response.text
    
"""assistant = InterviewAssistant()

@app.route('/upload_context', methods=['POST'])
def upload_context():
    data = request.json
    assistant.set_context(data['resume'], data['job_description'])
    return jsonify({"status": "success"})

@app.route('/generate_response', methods=['POST'])
def get_response():
    question = request.json['question']
    response = assistant.generate_response(question)
    return jsonify({"response": response})

@app.route('/generate_introduction', methods=['POST'])
def get_introduction():
    intro = assistant.generate_introduction()
    return jsonify({"introduction": intro})

# Audio handling
def audio_callback(indata, frames, time, status):
    # Process audio data here
    audio_data = np.mean(indata, axis=1)
    # Add speech-to-text processing here"""

if __name__ == '__main__':
    app.run(debug=True)