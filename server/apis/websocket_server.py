from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from threading import Lock
import json
from ml.fsl_svm_infer import FslSvmInfer, _SIGNING, _PREDICTING
from pydantic import BaseModel

router = APIRouter()
MODEL_PATH = "./models/fsl-svm-2-catv4.pkl"

# ── Model Manager ─────────────────────────────
class ModelManager:
    def __init__(self, path):
        self.lock = Lock()
        self.model_path = path
        self.infer = FslSvmInfer(path)

    def reload(self, new_path):
        with self.lock:
            self.model_path = new_path
            self.infer = FslSvmInfer(new_path)

    def get(self):
        return self.infer


model_manager = ModelManager(MODEL_PATH)

class ModelRequest(BaseModel):
    model:str

# PATCH: update model 
@router.patch("/model")
async def update_model(data: ModelRequest):
    model = data.model

    if model == "Model v1":
        model_manager.reload("./models/fsl-svm-2-catv7.pkl")
    elif model == "Model v2":
        model_manager.reload("./models/fsl-svm-2-catv8.pkl")
    elif model == "Model v3":
        model_manager.reload("./models/fsl-svm-2-catv9.pkl")

    return {
        "message": "model updated",
        "active_model": model
    }

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    prev_state = None

    try:
        while True:
            data = await websocket.receive_text()

            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Expected shape: { "landmarks": [[x,y,z] × 42] }
            landmarks = payload.get("landmarks")
            if landmarks is None or len(landmarks) != 42:
                continue

            infer = model_manager.get()

            prediction = infer.predict(landmarks)

            # Notify frontend when entering SIGNING state
            curr_state = infer.state
            if curr_state != prev_state:
                if curr_state == _SIGNING:
                    await websocket.send_text("__SIGNING__")
                prev_state = curr_state

            if prediction is not None:
                await websocket.send_text(prediction)
                prev_state = infer.state  # reset after prediction resets state

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
