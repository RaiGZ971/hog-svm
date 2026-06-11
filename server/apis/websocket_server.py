# from fastapi import APIRouter, WebSocket, WebSocketDisconnect
# import base64
# import numpy as np
# import cv2
# from ml.fsl_svm_infer import FslSvmInfer
#
# router = APIRouter()
#
# MODEL_PATH = "./models/fsl-svm-2-catv2.pkl"
#
#
# @router.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#
#     infer = FslSvmInfer(MODEL_PATH)
#
#     try:
#         while True:
#             data = await websocket.receive_text()
#
#             img_bytes = base64.b64decode(data)
#             np_arr = np.frombuffer(img_bytes, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#             if frame is None:
#                 continue
#
#             # run streaming inference
#             prediction = infer.predict(frame)
#
#             if prediction is not None:
#                 await websocket.send_text(prediction)
#
#     except WebSocketDisconnect:
#         pass
#     except Exception as e:
#         print(f"WebSocket error: {e}")

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import base64
import numpy as np
import cv2
from ml.fsl_svm_infer import FslSvmInfer, _SIGNING, _PREDICTING

router = APIRouter()
MODEL_PATH = "./models/fsl-svm-2-catv2.pkl"

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    infer = FslSvmInfer(MODEL_PATH)
    prev_state = infer.state

    try:
        while True:
            data      = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            prediction = infer.predict(frame)

            # notify frontend when state changes
            curr_state = infer.state
            if curr_state != prev_state:
                if curr_state == _SIGNING:
                    await websocket.send_text("__SIGNING__")
                prev_state = curr_state

            if prediction is not None:
                await websocket.send_text(prediction)
                prev_state = infer.state   # reset after prediction resets state

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


