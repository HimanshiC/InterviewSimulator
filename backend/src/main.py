
from fastapi import FastAPI, WebSocket, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import speech_recognition as sr
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI with your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple in-memory session storage
class InterviewSession:
    def __init__(self, resume: str, job_description: str):
        self.resume = resume
        self.job_description = job_description
        self.history: List[Dict] = []

active_sessions: Dict[str, InterviewSession] = {}

class JobDescription(BaseModel):
    text: str

# Initialize speech recognition
recognizer = sr.Recognizer()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Interview Simulator API is running"}

@app.post("/setup")
async def setup_interview(resume: UploadFile, job_description: JobDescription):
    """Set up a new interview session"""
    try:
        resume_content = await resume.read()
        session_id = str(len(active_sessions) + 1)  # Simple ID generation
        
        active_sessions[session_id] = InterviewSession(
            resume=resume_content.decode(),
            job_description=job_description.text
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Interview session created successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to setup interview: {str(e)}"
        }

async def generate_answer(question: str, session: InterviewSession) -> str:
    """Generate an answer using OpenAI"""
    try:
        messages = [
            {
                "role": "system",
                "content": f"""You are an AI interview assistant.
                Resume: {session.resume}
                Job Description: {session.job_description}
                
                Provide concise, professional responses based on the resume and job description.
                """
            },
            *[{"role": "user" if i % 2 == 0 else "assistant", "content": msg["content"]} 
              for i, msg in enumerate(session.history)],
            {"role": "user", "content": question}
        ]

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle real-time audio streaming and responses"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
        
    session = active_sessions[session_id]
    
    try:
        while True:
            # Receive audio data
            audio_data = await websocket.receive_bytes()
            
            try:
                # Convert audio to text
                audio = sr.AudioData(audio_data, sample_rate=44100, sample_width=2)
                text = recognizer.recognize_google(audio)
                
                # Generate response if it's a question
                if "?" in text:
                    answer = await generate_answer(text, session)
                    
                    # Update session history
                    session.history.append({"content": text})
                    session.history.append({"content": answer})
                    
                    await websocket.send_json({
                        "type": "answer",
                        "question": text,
                        "answer": answer
                    })
                else:
                    # Just send back transcription
                    await websocket.send_json({
                        "type": "transcription",
                        "text": text
                    })
                    
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)*/