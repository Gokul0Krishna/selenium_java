from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Simulated background task
def background_task(task_name: str):
    print(f"[BACKGROUND] Starting {task_name}...")
    time.sleep(5)
    print(f"[BACKGROUND] Completed {task_name}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run_task")
async def run_task(data: dict, background_tasks: BackgroundTasks):
    task_name = data.get("task", "unknown")
    background_tasks.add_task(background_task, task_name)
    return JSONResponse({"message": f"✅ {task_name} started in background!"})
    