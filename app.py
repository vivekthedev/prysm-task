from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from main import run_chatbot
from file_mappings import mapping
from enum import Enum

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class StockSymbol(str, Enum):
    ETERNAL = "ETERNAL.NS"
    SWIGGY = "SWIGGY.NS"
    TCS = "TCS.NS"

class ChatRequest(BaseModel):
    query: str
    symbol: StockSymbol
    documents: list[str] = []

@app.get("/form", response_class=HTMLResponse)
async def chat_form(request: Request):
    symbols = [
        {"value": "ETERNAL.NS"},
        {"value": "SWIGGY.NS"},
        {"value": "TCS.NS"}
    ]
    
    documents = [
        {"value": "document 1", "label": "document 1"},
        {"value": "document 2", "label": "document 2"},
        {"value": "document 3", "label": "document 3"},
    ]
    
    return templates.TemplateResponse("chat_form.html", {
        "request": request,
        "symbols": symbols,
        "documents": documents
    })

@app.get("/")
async def healthcheck():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatRequest):
    ticker = request.symbol.value
    documents = []
    base = mapping[ticker]
    collection = base['parent']
    if request.documents:
        documents = [f"documents/{collection}/{base[doc]}" for doc in request.documents]
    response = run_chatbot(request.query, ticker, documents, collection)
    return {
        "response": response,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)