from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Load POI data saat server start
with open("poi_data.json", "r", encoding="utf-8") as f:
    POI_LIST = json.load(f)

# Buat lookup dict by id untuk cari posisi cepat
POI_MAP = {poi["id"]: poi for poi in POI_LIST}


def build_system_prompt() -> str:
    lines = [
        "You are a navigation assistant for PSDKU Lamongan campus indoor navigation.",
        "Identify which room the user wants to go to based on their input.",
        "",
        "Available rooms:"
    ]
    for poi in POI_LIST:
        aliases = ", ".join(poi["alias"])
        lines.append(f'- id: "{poi["id"]}", nama: "{poi["nama"]}", alias: [{aliases}]')
    
    lines += [
        "",
        "Rules:",
        '- Return ONLY the exact id from the list above',
        '- If not found or unclear, return "unknown"',
        "- No explanation, just the id"
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = build_system_prompt()


class NavigateRequest(BaseModel):
    text: str


class NavigateResponse(BaseModel):
    poi_id: str
    nama: str
    posisi: dict | None


@app.get("/")
def root():
    return {"status": "WholePSDKU Backend running"}


@app.get("/pois")
def get_all_pois():
    return POI_LIST


@app.post("/navigate", response_model=NavigateResponse)
async def navigate(req: NavigateRequest):
    full_prompt = f'{SYSTEM_PROMPT}\n\nUser input: "{req.text}"\n\nRoom id:'

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama tidak bisa diakses: {str(e)}")

    result = response.json()
    poi_id = result["response"].strip().lower().replace('"', '').replace("'", "")

    if poi_id == "unknown" or poi_id not in POI_MAP:
        return NavigateResponse(poi_id="unknown", nama="Tidak ditemukan", posisi=None)

    target = POI_MAP[poi_id]
    return NavigateResponse(
        poi_id=poi_id,
        nama=target["nama"],
        posisi=target["posisi"]
    )