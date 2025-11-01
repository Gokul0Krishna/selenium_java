from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
from reg import Regression
app = FastAPI()
reg=Regression()
templates = Jinja2Templates(directory=r"C:\Users\ASUS\OneDrive\Desktop\code\cp\javaproject\year3\frontend\templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/data_process")
async def run_task(data: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(reg.load_transform_data(),'regression' )
    return JSONResponse({"message": f"✅ regression started in background!"})