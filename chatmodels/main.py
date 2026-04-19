from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from chatmodels.swasthya_mitra import (
    get_ai_response,
    get_sensor_response,
    reset_chat_history,
    initialize_vector_db
)

app = FastAPI()

class ChatRequest(BaseModel):
    email: str
    message: Optional[str] = None 
    heart_rate: Optional[int] = None
    temperature: Optional[float] = None
    ecg: Optional[str] = None
    user_profile: Optional[dict] = None 

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.on_event("startup")
async def startup_event():
    initialize_vector_db()

@app.post("/chat")
async def chat(req: ChatRequest):

    user_id = req.email

    try:
        if req.message is None:
            if req.heart_rate is None and req.temperature is None and req.ecg is None:
                return {"error": "No sensor data provided"}
            reply = get_sensor_response(
                req.heart_rate,
                req.temperature,
                req.ecg,
                user_id,
                req.user_profile
            )
            return {"reply": reply}

        reply = get_ai_response(
            req.message,
            req.heart_rate,
            req.temperature,
            req.ecg,
            user_id,
            req.user_profile
        )

        return {"reply": reply}

    except Exception as e:
        return {"error": str(e)}
    
@app.post("/reset")
async def reset(req: ChatRequest):
    reset_chat_history(req.email)
    return {"message": "Chat history cleared"}