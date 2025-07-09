from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
import uuid
import os

from evaluation.vllm_wrapper import human_reasoning_wrapper, websocket_human_reasoning_wrapper, AsyncVLLMClient
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
active_sessions = {}

# Initialize models (adapt from runner.py)
@app.on_event("startup")
async def startup_event():
    global vsl_model, txt_model, txt_sampling_params, vsl_sampling_params
    # Vision model
    vsl_model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    vsl_model = AsyncVLLMClient(model_path=vsl_model_path, port=41651)
    # Human reasoning wrapper over WebSocket (injected per session)
    # Text sampling params
    txt_sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=512)
    # Vision sampling params
    vsl_sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=128)

@app.post("/api/start_session")
async def start_session(session_data: ChatSession):
    """Start a new human reasoning session"""
    session_id = str(uuid.uuid4())
    txt_msg, vsl_msg = initialize_msg(
        session_data.question,
        session_data.image_path,
        encode_images=True
    )
    active_sessions[session_id] = {
        "txt_msg": txt_msg,
        "vsl_msg": vsl_msg,
        "ground_truth": session_data.ground_truth,
        "conversation_history": [],
        "rounds": 0,
        "status": "active"
    }
    return {"session_id": session_id, "question": session_data.question}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat using original pipeline"""
    await websocket.accept()
    if session_id not in active_sessions:
        await websocket.send_json({"error": "Session not found"})
        return

    session = active_sessions[session_id]

    # Create a WebSocket–integrated human reasoning wrapper
    txt_model = websocket_human_reasoning_wrapper(websocket)
    # Retrieve shared vision model and sampling params
    global vsl_model, txt_sampling_params, vsl_sampling_params

    try:
        # Loop until user signals final answer or disconnects
        while True:
            # Receive human reasoning input
            data = await websocket.receive_json()
            human_input = data.get("message", "")
            if human_input.lower() in ["quit", "exit"]:
                break

            session["rounds"] += 1

            # Append human turn to text message history
            session["txt_msg"].append({"role": "user", "content": human_input})

            # Invoke the original batch_process_conversations for one round
            results = batch_process_conversations(
                [session["txt_msg"]],
                [session["vsl_msg"]],
                txt_model,
                vsl_model,
                txt_sampling_params,
                vsl_sampling_params,
                max_rounds=session["rounds"],
                txt_batch_size=1,
                vis_batch_size=1,
                timeout=0.1,
                conv_round_prompt=False
            )

            # Extract the latest vision response
            vision_resp = results[0][-1]["vision_response"]

            # Send vision response back to frontend
            await websocket.send_json({
                "type": "vision_response",
                "content": vision_resp,
                "round": session["rounds"]
            })

            # Update session history
            session["conversation_history"].extend([
                {"role": "human", "content": human_input},
                {"role": "vision", "content": vision_resp}
            ])

            # Check for final answer pattern
            if "answer is:" in human_input.lower():
                import re
                pattern = r"(?:The|My) answer is:\s*\(([A-D])\)"
                match = re.search(pattern, human_input, re.IGNORECASE)
                if match:
                    final_answer = match.group(1)
                    is_correct = (final_answer == session["ground_truth"].strip("()"))
                    await websocket.send_json({
                        "type": "final_answer",
                        "answer": final_answer,
                        "ground_truth": session["ground_truth"],
                        "correct": is_correct,
                        "rounds": session["rounds"]
                    })
                    break

    except WebSocketDisconnect:
        session["status"] = "disconnected"

@app.get("/api/datasets")
async def get_available_datasets():
    """Return available datasets for selection"""
    return {
        "spubench_500": {
            "name": "MM-SpuBench (500 samples)",
            "description": "Spurious correlation benchmark"
        }
    }

@app.get("/api/dataset/{dataset_name}/samples")
async def get_dataset_samples(dataset_name: str, limit: int = 10):
    """Get samples from a specific dataset"""
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
    for sample in samples[:limit]:
        sample["path"] = os.path.join(test_sets[dataset_name]["img_dir"], sample["path"])
    return {"samples": samples[:limit]}

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="web_ui", html=True), name="static")
