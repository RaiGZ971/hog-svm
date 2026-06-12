from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from ml.fsl_svm_infer import FslSvmInfer, _SIGNING, _PREDICTING

router = APIRouter()
MODEL_PATH = "./models/fsl-svm-2-catv4.pkl"


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    infer = FslSvmInfer(MODEL_PATH)
    prev_state = infer.state

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
