from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
from fsl_svm_model import FslSvm

app = FastAPI()

model = FslSvm()
model.load_svm_model("svm_hog_model.pkl")


@app.websocket("/ws/infer")
async def infer_socket(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            if request["action"] == "predict":
                frame = request["frame"]  # base64 image

                result = model.predict_frame(frame)

                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "label": result["label"],
                    "confidence": result["confidence"]
                }))

            else:
                await websocket.send_text(json.dumps({
                    "error": "Unknown action"
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
