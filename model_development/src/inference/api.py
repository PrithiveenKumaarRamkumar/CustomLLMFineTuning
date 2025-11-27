from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import os

app = FastAPI(title="StarCoder2 Inference API")

MODEL = None
TOKENIZER = None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 150
    temperature: float = 0.7
    do_sample: bool = True

class GenerateResponse(BaseModel):
    generated_text: str
    inference_time: float

@app.on_event("startup")
async def load_model():
    global MODEL, TOKENIZER
    
    BASE_MODEL = os.getenv("BASE_MODEL", "bigcode/starcoder2-3b")
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/checkpoints/starcoder-finetuned")
    
    print(f"Loading model from {MODEL_PATH}...")
    
    TOKENIZER = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
    TOKENIZER.pad_token = TOKENIZER.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    MODEL = PeftModel.from_pretrained(base_model, MODEL_PATH)
    MODEL = MODEL.merge_and_unload()
    MODEL.eval()
    
    print("✓ Model loaded")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    inputs = TOKENIZER(request.prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs, max_length=request.max_length,
            temperature=request.temperature, do_sample=request.do_sample,
            pad_token_id=TOKENIZER.eos_token_id
        )
    
    text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return GenerateResponse(generated_text=text, inference_time=time.time() - start)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)