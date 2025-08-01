import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import speech_recognition as sr
import threading
import wave
import pyaudio
import wave
import threading
import queue
import time
from datetime import datetime
import numpy as np
import noisereduce as nr
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please set GOOGLE_API_KEY environment variable")

print(f"Initializing Gemini with API key:")
genai.configure(api_key=api_key)

appa = Flask(__name__)
CORS(appa)

# Configure Gemini api
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#model = genai.GenerativeModel('gemini-pro')
model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
chat = model.start_chat(history=[])

# Store context globally to access a number of methods

context = {
    "resume": "",
    "job_description": ""
}

class AudioProcessor:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.noise_sample = None
        
    def reduce_noise(self, audio_chunk):
        # Convert audio chunk to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        # If we have a noise profile, use it for reduction
        if self.noise_sample is not None:
            reduced = nr.reduce_noise(
                y=audio_data.astype(float),
                sr=self.RATE,
                prop_decrease=0.95,
                stationary=True
            )
            return reduced.astype(np.int16).tobytes()
        return audio_chunk
        
    def calibrate_noise(self, duration=2):
        """Record ambient noise for noise reduction calibration"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                       channels=self.CHANNELS,
                       rate=self.RATE,
                       input=True,
                       frames_per_buffer=self.CHUNK)
        
        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * duration)):
            data = stream.read(self.CHUNK)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Convert to numpy array for noise profile
        self.noise_sample = np.frombuffer(b''.join(frames), dtype=np.int16)

class QuestionDetector:
    def __init__(self):
        self.question_starters = [
            'what', 'why', 'when', 'where', 'how', 'could', 'can', 'would',
            'tell me', 'describe', 'explain', 'do you', 'have you'
        ]
        
    def is_question(self, text):
        text = text.lower().strip()
        # Check for question marks
        if '?' in text:
            return True
        # Check for question starters
        return any(text.startswith(starter) for starter in self.question_starters)
        
    def extract_questions(self, text):
        sentences = sent_tokenize(text)
        questions = [s for s in sentences if self.is_question(s)]
        return questions

class InterviewAssistant:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.question_detector = QuestionDetector()
        self.is_recording = False
        self.frames = []
        self.audio_queue = queue.Queue()
        self.recognizer = sr.Recognizer()
        self.stealth_mode = False
        
    def toggle_stealth_mode(self, enabled):
        self.stealth_mode = enabled
        return {"status": "Stealth mode " + ("enabled" if enabled else "disabled")}
        
    # def start_recording(self):
    #     # Calibrate noise reduction first
    #     print("Calibrating noise reduction...")
    #     self.audio_processor.calibrate_noise()
        
    #     self.is_recording = True
    #     threading.Thread(target=self._record_audio).start()
    #     threading.Thread(target=self._process_audio).start()
    #     return transcribed_text, gemini_response
    def start_recording(self):
    # Calibrate noise reduction
        print("Calibrating noise reduction...")
        self.audio_processor.calibrate_noise()

        self.is_recording = True
        self.frames = []

    # Record synchronously for a few seconds
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_processor.FORMAT,
                    channels=self.audio_processor.CHANNELS,
                    rate=self.audio_processor.RATE,
                    input=True,
                    frames_per_buffer=self.audio_processor.CHUNK)

        print("Recording...")
        for _ in range(0, int(self.audio_processor.RATE / self.audio_processor.CHUNK * 5)):  # ~5 seconds
            data = stream.read(self.audio_processor.CHUNK)
            reduced_data = self.audio_processor.reduce_noise(data)
            self.frames.append(reduced_data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    # Save the recorded frames to a WAV file
        filename = "temp.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.audio_processor.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.audio_processor.FORMAT))
        wf.setframerate(self.audio_processor.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    # Transcribe and send to Gemini
        with sr.AudioFile(filename) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"[USER]: {text}")
                if self.question_detector.is_question(text):
                    print("[INFO] Detected question.")
                    response = self.question_detector.get_answer(text)
                    print(f"[GEMINI]: {response}")
                else:
                    response = "No question detected."
            except sr.UnknownValueError:
                print("[WARN] Google Speech could not understand audio")
                text = ""
                response = "Sorry, I couldn’t understand you."

        return text, response

    
    def stop_recording(self):
        self.is_recording = False
        
    def _process_audio(self):
        while self.is_recording:
            if not self.audio_queue.empty():
                frames = self.audio_queue.get()

            # Save to temp WAV file
                filename = "temp_audio.wav"
                wf = wave.open(filename, 'wb')
                wf.setnchannels(self.audio_processor.CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_processor.FORMAT))
                wf.setframerate(self.audio_processor.RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

            # Recognize using speech_recognition
                with sr.AudioFile(filename) as source:
                    audio = self.recognizer.record(source)
                    try:
                        text = self.recognizer.recognize_google(audio)
                        print(f"[USER]: {text}")

                    # Optional: is it a question?
                        if self.question_detector.is_question(text):
                            print("[INFO] Detected question.")
                    
                    # Send to Gemini
                        prompt = f"""
                        Based on this resume:
                        {context['resume']}

                        And this job description:
                        {context['job_description']}

                        Please provide a short, professional answer to this question:
                        {text}
                        """
                        response = chat.send_message(prompt, stream=True)
                        response_text = ' '.join([chunk.text for chunk in response])
                        print(f"[GEMINI]: {response_text}")

                    except sr.UnknownValueError:
                        print("[WARN] Google Speech could not understand audio")
                    except sr.RequestError as e:
                        print(f"[ERROR] Could not request results; {e}")
            time.sleep(0.5)



    def _record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_processor.FORMAT,
                       channels=self.audio_processor.CHANNELS,
                       rate=self.audio_processor.RATE,
                       input=True,
                       frames_per_buffer=self.audio_processor.CHUNK)
        
        while self.is_recording:
            data = stream.read(self.audio_processor.CHUNK)
            # Apply noise reduction
            reduced_data = self.audio_processor.reduce_noise(data)
            self.frames.append(reduced_data)
            
            if len(self.frames) >= (self.audio_processor.RATE / self.audio_processor.CHUNK * 5):
                self.audio_queue.put(self.frames[:])
                self.frames = []
        
        stream.stop_stream()
        stream.close()
        p.terminate()


assistant = InterviewAssistant()
audio_handler = AudioProcessor()
@appa.route('/')
def home():
    return render_template('indexa.html')

@appa.route('/upload_context', methods=['POST'])
def upload_context():
    data = request.json
    context["resume"] = data.get('resume', '')
    context["job_description"] = data.get('jobDescription', '')
    return jsonify({"status": "success", "message": "Context uploaded successfully"})


@appa.route('/start_recording', methods=['POST'])
def start_recording():
    audio_handler.start_recording()
    return jsonify({"status": "Recording started"})

@appa.route('/stop_recording', methods=['POST'])
def stop_recording():
    audio_handler.stop_recording()
    return jsonify({"status": "Recording stopped"})

@appa.route('/generate_response', methods=['POST'])
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
@appa.route('/generate_followup', methods=['POST'])
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
        
        Please provide a professional followup response to this interview question giving weightage to the question and ensure that it is short and relevant and sounds human:
        {question}
        """
        
        response1 = chat.send_message(prompt, stream=True)
        response_text1 = ' '.join([chunk.text for chunk in response1])
        
        return jsonify({"response": response_text1})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@appa.route('/start_listening', methods=['POST'])
def start_listening():
    #text = assistant.start_listening()
    #assistant.start_recording()
    #return jsonify({"text": "Listening started, recording audio now!"})
    text, response = assistant.start_recording()
    return jsonify({
        "text": text,
        "response": response
    })
    #return jsonify({"text": text})

#@appa.route('/generate_introduction', methods=['POST'])
@appa.route('/generate_introduction', methods=['POST'])
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
@appa.route('/toggle_stealth', methods=['POST'])
def toggle_stealth():
    data = request.json
    enabled = data.get('enabled', False)
    return jsonify(assistant.toggle_stealth_mode(enabled))
if __name__ == '__main__':
    appa.run(debug=True)