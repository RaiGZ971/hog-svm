from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from inference.pipeline import Pipeline
import cv2
import numpy as np

app = FastAPI()
pipeline = Pipeline()

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()

            # ignore text frames (ping, handshake artifacts, etc.)
            if "bytes" not in message or message["bytes"] is None:
                continue

            data = message["bytes"]
            img = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )

            if img is None:
                await websocket.send_json({"error": "invalid image"})
                continue

            pred = pipeline.predict(img)
            await websocket.send_json({"prediction": pred})

    except WebSocketDisconnect:
        pass
