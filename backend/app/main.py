from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message":"API Working"}

@app.post("/login")
def login():
    return {"message":"Coming Soon"}

@app.post("/register")
def register():
    return {"message":"Coming Soon"}

@app.post("/chat")
def chat():
    return {"message":"Coming Soon"}

@app.get("/dashboard")
def dashboard():
    return {"message":"Coming Soon"}

@app.get("/history")
def history():
    return {"message":"Coming Soon"}