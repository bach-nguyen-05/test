from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
from typing import List, Dict, Any
import uuid
import os

# Import your existing modules
from evaluation.vllm_wrapper import human_reasoning_wrapper, AsyncVLLMClient
from shared.conv_sampling import batch_process_conversations
from shared.utils import initialize_msg
import vllm

app = FastAPI(title="Human Reasoning Web UI")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ChatSession(BaseModel):
    session_id: str
    question: str
    image_path: str
    ground_truth: str = None

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str = None

# Global storage for active sessions
active_sessions: Dict[str, Dict] = {}

# Initialize models (adapt from your runner.py)
def initialize_models():
    global vsl_model
    vsl_model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    vsl_model = AsyncVLLMClient(model_path=vsl_model_path, port=41651)
    return vsl_model

@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.post("/api/start_session")
async def start_session(session_data: ChatSession):
    """Start a new human reasoning session"""
    session_id = str(uuid.uuid4())
    
    # Initialize messages using your existing function
    txt_msg, vsl_msg = initialize_msg(
        session_data.question, 
        session_data.image_path, 
        encode_images=True
    )
    
    active_sessions[session_id] = {
        "txt_msg": txt_msg,
        "vsl_msg": vsl_msg,
        "question": session_data.question,
        "image_path": session_data.image_path,
        "ground_truth": session_data.ground_truth,
        "conversation_history": [],
        "rounds": 0,
        "status": "active"
    }
    
    return {"session_id": session_id, "question": session_data.question}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_json({"error": "Session not found"})
        return
    
    session = active_sessions[session_id]
    
    try:
        while True:
            # Receive human input
            data = await websocket.receive_json()
            human_response = data.get("message", "")
            
            if human_response.lower() in ['quit', 'exit']:
                break
            
            # Process with vision model
            session["rounds"] += 1
            
            # Check for final answer
            if "answer is:" in human_response.lower():
                # Extract and validate final answer
                import re
                pattern = r"(?:The|My) answer is:\s*\(([A-D])\)"
                match = re.search(pattern, human_response, re.IGNORECASE)
                
                if match:
                    final_answer = match.group(1)
                    is_correct = final_answer == session["ground_truth"].strip("()")
                    
                    await websocket.send_json({
                        "type": "final_answer",
                        "answer": final_answer,
                        "ground_truth": session["ground_truth"],
                        "correct": is_correct,
                        "rounds": session["rounds"]
                    })
                    break
            
            # Send to vision model
            vsl_history = session["vsl_msg"].copy()
            vsl_history.append({"role": "user", "content": human_response})
            
            vsl_sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=128)
            vision_response = await vsl_model.predict([vsl_history], vsl_sampling_params)
            
            # Send vision response back to frontend
            await websocket.send_json({
                "type": "vision_response",
                "content": vision_response[0],
                "round": session["rounds"]
            })
            
            # Update session history
            session["conversation_history"].extend([
                {"role": "human", "content": human_response},
                {"role": "vision", "content": vision_response[0]}
            ])
            
    except WebSocketDisconnect:
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "disconnected"

@app.get("/api/datasets")
async def get_available_datasets():
    """Return available datasets for selection"""
    datasets = {
        "spubench_500": {
            "name": "MM-SpuBench (500 samples)",
            "description": "Spurious correlation benchmark"
        }
    }
    return datasets

@app.get("/api/dataset/{dataset_name}/samples")
async def get_dataset_samples(dataset_name: str, limit: int = 10):
    """Get samples from a specific dataset"""
    # Load samples from your existing dataset files
    test_sets = {
        "spubench_500": {
            "test_file": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench/data.json",
            "img_dir": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench"
        }
    }
    
    if dataset_name not in test_sets:
        return {"error": "Dataset not found"}
    
    with open(test_sets[dataset_name]["test_file"], "r") as f:
        samples = json.load(f)
    
    # Add full image paths and limit samples
    for sample in samples[:limit]:
        sample["path"] = os.path.join(test_sets[dataset_name]["img_dir"], sample["path"])
    
    return {"samples": samples[:limit]}

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="web_ui", html=True), name="static")
