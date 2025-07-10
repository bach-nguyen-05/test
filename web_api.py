from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
import uuid
import os
from queue import Queue, Empty
import vllm

# Import your existing modules
from evaluation.vllm_wrapper import human_reasoning_wrapper, AsyncVLLMClient
from shared.conv_sampling import batch_process_conversations
from shared.utils import initialize_msg

#########################################
app = FastAPI(title="Human Reasoning Web UI")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
active_sessions = {}

########################################

# Global models
vsl_model = None
txt_sampling_params = None
vsl_sampling_params = None

@app.on_event("startup")
async def startup_event():
    """Initialize models like runner.py"""
    global vsl_model, txt_sampling_params, vsl_sampling_params
    
    # Initialize vision model
    vsl_model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    vsl_model = AsyncVLLMClient(
        model_path=vsl_model_path, 
        port=41651
    )
    
    # sampling parameters (matching runner.py)
    txt_sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=512)
    vsl_sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=128)

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
    """WebSocket endpoint integrated with batch_process_conversations"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_json({"error": "Session not found"})
        return
    
    session = active_sessions[session_id]
    
    # Create WebSocket-enabled human reasoning wrapper
    txt_model = human_reasoning_wrapper(websocket=websocket)
    
    # Background task to run batch_process_conversations
    async def run_conversation():
        """Run the original pipeline in background thread"""
        loop = asyncio.get_running_loop()

        ### Same as results = batch_process_conversations(...) in runner.py ###
        return await loop.run_in_executor(
            None,
            # SAME PIPELINE AS RUNNER.PY
            batch_process_conversations,
            [session["txt_msg"]],
            [session["vsl_msg"]],
            txt_model,
            vsl_model,
            txt_sampling_params,
            vsl_sampling_params,
            12,  
            1,   # txt_batch_size
            1,   # vis_batch_size
            0.1, # timeout
            False # conv_round_prompt
        )
    
    # Start the pipeline
    conversation_task = asyncio.create_task(run_conversation())
    
    try:
        while True:
            # Handle outgoing requests from human_reasoning_wrapper
            try:
                req = txt_model.req_q.get_nowait()
                await websocket.send_json(req)
            except Empty:
                pass
            
            # Handle incoming responses from frontend
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                human_response = data.get("message", "")
                
                if human_response.lower() in ['quit', 'exit']:
                    break
                
                # Send response back to wrapper
                txt_model.res_q.put(human_response)
                
                # Counting Rounds
                session["rounds"] += 1
                session["conversation_history"].append({
                    "role": "human", 
                    "content": human_response,
                    "round": session["rounds"]
                })
                
                # Check if this is a final answer (no longer needs "The answer is:" format)
                if any(letter in human_response.upper() for letter in ['A', 'B', 'C', 'D']) and \
                   len(human_response.strip()) <= 10:  # Simple final answer detection
                    # Extract answer letter
                    import re
                    match = re.search(r'[A-D]', human_response.upper())
                    if match:
                        final_answer = match.group(0)
                        is_correct = final_answer == session["ground_truth"].strip("()")
                        
                        await websocket.send_json({
                            "type": "final_answer",
                            "answer": final_answer,
                            "ground_truth": session["ground_truth"],
                            "correct": is_correct,
                            "rounds": session["rounds"]
                        })
                        break
                        
            except asyncio.TimeoutError:
                continue
                
    except WebSocketDisconnect:
        session["status"] = "disconnected"
    finally:
        conversation_task.cancel()

@app.get("/api/datasets")
async def get_available_datasets():
    """Return available datasets for selection"""
    return {
        "spubench_500": {
            "name": "SpuBench (500 samples)",
            "description": "Sample Benchmark"
        }
    }

@app.get("/api/dataset/{dataset_name}/samples")
async def get_dataset_samples(dataset_name: str, limit: int = 10):
    ### Same as runner.py but specific dataset###
    test_sets = {
        "spubench_500": {
            "test_file": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench/data.json",
            "img_dir": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench"
        }
    }
    
    if dataset_name not in test_sets:
        return {"error": "Dataset not found"}
    
    try:
        with open(test_sets[dataset_name]["test_file"], "r") as f:
            samples = json.load(f)
        
        for sample in samples[:limit]:
            sample["path"] = os.path.join(test_sets[dataset_name]["img_dir"], sample["path"])
        
        return {"samples": samples[:limit]}
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="web_ui", html=True), name="static")
