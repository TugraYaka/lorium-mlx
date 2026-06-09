import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load, generate, stream_generate

app = FastAPI(title="Lorium MLX Engine")

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load_model(self, model_path: str):
        print(f"Loading model from {model_path}...", flush=True)
        self.model, self.tokenizer = load(model_path)
        self.model_name = model_path
        print("Model loaded successfully.", flush=True)

manager = ModelManager()

class LoadRequest(BaseModel):
    model_path: str

from typing import List, Optional

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[dict]] = None
    max_tokens: int = 4096
    temperature: float = 0.7

@app.get("/status")
def status():
    return {
        "status": "online",
        "engine": "mlx",
        "loaded_model": manager.model_name
    }

@app.post("/load")
def load_model(req: LoadRequest):
    try:
        manager.load_model(req.model_path)
        return {"success": True, "message": f"Loaded {req.model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def prepare_prompt(req: GenerateRequest) -> str:
    if hasattr(manager.tokenizer, "apply_chat_template") and manager.tokenizer.chat_template:
        if req.messages and len(req.messages) > 0:
            return manager.tokenizer.apply_chat_template(
                req.messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif req.prompt:
            return manager.tokenizer.apply_chat_template(
                [{"role": "user", "content": req.prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
    return req.prompt if req.prompt else ""

@app.post("/generate")
def generate_text(req: GenerateRequest):
    if not manager.model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        formatted_prompt = prepare_prompt(req)
        response = generate(manager.model, manager.tokenizer, prompt=formatted_prompt, max_tokens=req.max_tokens, verbose=True)
        return {"text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
def stream_text(req: GenerateRequest):
    if not manager.model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        formatted_prompt = prepare_prompt(req)

        def event_generator():
            for response in stream_generate(manager.model, manager.tokenizer, prompt=formatted_prompt, max_tokens=req.max_tokens):
                yield response.text

        return StreamingResponse(event_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    print(f"Starting MLX Engine on port {port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
