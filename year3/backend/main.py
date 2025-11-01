from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
from reg import Regression
from classi import Classification

traindl,valdl,testdl=None,None,None

app = FastAPI()
reg=Regression()
clas=Classification()
templates = Jinja2Templates(directory=r"C:\Users\ASUS\OneDrive\Desktop\code\cp\javaproject\year3\frontend\templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/data_process")
async def run_task(data: dict, background_tasks: BackgroundTasks):
    global traindl,testdl,valdl
    task_name = data.get("task", "unknown")
    if task_name == "regression":
        background_tasks.add_task(reg.load_transform_data(),task_name)
        traindl,testdl,valdl = reg.load_transform_data()
    elif task_name == "classification":
        background_tasks.add_task(clas.load_transform_data(),task_name )
        traindl,testdl,valdl = clas.load_transform_data()
    return JSONResponse({"message": f"✅{task_name}  completed in background!"})