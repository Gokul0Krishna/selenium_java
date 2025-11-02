from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
from reg import Regression
from classi import Classification

traindl,valdl,testdl,taskname=None,None,None,None

app = FastAPI()
reg=Regression()
clas=Classification()
templates = Jinja2Templates(directory=r"C:\Users\ASUS\OneDrive\Desktop\code\cp\javaproject\year3\frontend\templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/data_process")
async def run_task(data: dict, background_tasks: BackgroundTasks):
    global traindl,testdl,valdl,taskname
    task_name = data.get("task", "unknown")
    taskname = task_name
    if task_name == "regression":
        background_tasks.add_task(reg.load_transform_data(),task_name)
        traindl,testdl,valdl = reg.load_transform_data()
    elif task_name == "classification":
        background_tasks.add_task(clas.load_transform_data(),task_name )
        traindl,testdl,valdl = clas.load_transform_data()
    return JSONResponse({"message": f"✅{task_name}  completed in background!"})

@app.post("/train")
async def train_model(data: dict, background_tasks: BackgroundTasks):
    global traindl,testdl,valdl
    epochs = int(data.get("epochs", 10))
    lr = float(data.get("lr", 0.01))
    if taskname == 'regression':
        train_acc,train_loss,val_acc,val_loss=reg.train(traindl=traindl,valdl=valdl,epoches=epochs,lr=lr)
