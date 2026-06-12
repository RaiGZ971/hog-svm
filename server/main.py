from fastapi import FastAPI
from apis.websocket_server import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow frontend (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
