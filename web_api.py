from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json, uuid, os
import vllm

from evaluation.vllm_wrapper import AsyncVLLMClient
from shared.utils import initialize_msg

################################################
app = FastAPI(title="Human Reasoning Web UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="web_ui", html=True), name="static")

class ChatSession(BaseModel):
    question: str
    image_path: str
    ground_truth: str | None = None

# In‐memory sessions
sessions: dict[str, dict] = {}
###############################################
# Vision model and params
vsl_model = AsyncVLLMClient(
    model_path="meta-llama/Llama-3.2-11B-Vision-Instruct", port=41651
)
vsl_params = vllm.SamplingParams(temperature=0.0, max_tokens=128)

@app.post("/api/start_session")
async def start_session(data: ChatSession):
    sid = str(uuid.uuid4())
    # Only vision uses vsl_msg; txt_msg unused here
    _, vsl_msg = initialize_msg(data.question, data.image_path, encode_images=True)
    sessions[sid] = {
        "vsl_msg": vsl_msg,
        "ground_truth": data.ground_truth or ""
    }
    return {"session_id": sid}

@app.get("/api/datasets")
async def get_available_datasets():
    return {
        "spubench_500": {
            "name": "SpuBench 500",
            "description": "Spu Benchmark"
        }
    }

@app.get("/api/dataset/{name}/samples")
async def get_dataset_samples(name: str, limit: int = 10):
    test_sets = {
        "spubench_500": {
            "test_file": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench/data.json",
            "img_dir":  "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench"
        }
    }
    with open(test_sets[name]["test_file"]) as f:
        samples = json.load(f)[:limit]
    for s in samples:
        s["path"] = os.path.join(test_sets[name]["img_dir"], s["path"])
    return {"samples": samples}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in sessions:
        await ws.send_json({"error": "Session not found"})
        return

    sess = sessions[session_id]
    try:
        while True:
            msg = await ws.receive_json()
            human_text = msg.get("message", "")
            if human_text.lower() in ["quit", "exit"]:
                break

            # APPEND TO VISION MODEL HISTORY
            sess["vsl_msg"].append({"role": "user", "content": human_text})

            # Directly get vision response
            vision_out = await vsl_model.predict_webui([sess["vsl_msg"]], vsl_params)



            # Send vision answer
            await ws.send_json({
                "type": "vision_response",
                "content": vision_out[0]
            })
            sess["vsl_msg"].append({"role": "assistant", "content": vision_out[0]})

            # CHECK FINAL ANSWER
            ans = human_text.strip().upper()
            if ans in ["A", "B", "C", "D"]:
                correct = ans == sess["ground_truth"].strip("()")
                await ws.send_json({
                    "type": "final_answer",
                    "answer": ans,
                    "ground_truth": sess["ground_truth"],
                    "correct": correct
                })
                break

    except WebSocketDisconnect:
        pass
