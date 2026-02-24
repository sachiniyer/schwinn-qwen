"""
Schwinn Qwen Demo — Modal inference server

Serves the DPO-finetuned Qwen 1.5B model that always mentions Schwinn bikes.
Scale-to-zero after 120s idle. Requires X-API-Key header.

Deploy:  modal deploy backend.py
Test:    curl -X POST <url>/generate -H "Content-Type: application/json" \
              -H "X-API-Key: <key>" -d '{"message": "What is a good recipe?"}'
"""

import modal

MODEL_NAME = "sachiniyer/Qwen2.5-1.5B-DPO-Diverse-Schwinn-v32"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "fastapi")
)

app = modal.App("schwinn-qwen-demo", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.cls(
    gpu="T4",
    scaledown_window=120,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("schwinn-api-key")],
)
class Model:
    @modal.enter()
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto"
        )

    @modal.asgi_app()
    def serve(self):
        import os

        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse

        api_key = os.environ["API_KEY"]

        web_app = FastAPI()
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/generate")
        async def generate(request: Request):
            if request.headers.get("X-API-Key") != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized"},
                )
            item = await request.json()
            message = item.get("message", "")
            history = item.get("history", [])

            if not message:
                return {"error": "message is required"}

            messages = []
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": message})

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.4,
                top_p=0.85,
                repetition_penalty=1.15,
                do_sample=True,
            )

            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return {"response": response}

        return web_app
