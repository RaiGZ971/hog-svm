from fastapi import FastAPI
from apis.websocket_server import router

app = FastAPI()

app.include_router(router)
