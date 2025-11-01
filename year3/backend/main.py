from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import FileResponse
from reg import Regression
import os

app = FastAPI()
reg=Regression()

# Allow frontend JS to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = r"C:\Users\ASUS\OneDrive\Desktop\code\cp\javaproject\year3\frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

class TrainRequest(BaseModel):
    action: str

@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/train")
def train_endpoint(req: TrainRequest):
    traindl,testdl,valdl = reg.load_transform_data()
    print(traindl[:5])