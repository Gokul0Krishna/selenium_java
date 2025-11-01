from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow frontend JS to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    action: str

@app.post("/train")
def train_endpoint(req: TrainRequest):
    # final_loss = train_model(req.action)
    # return {"action": req.action, "final_loss": final_loss}
    pass