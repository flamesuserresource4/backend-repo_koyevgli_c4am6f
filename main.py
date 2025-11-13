import os
import io
import zipfile
import asyncio
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

# ----- FastAPI app -----
app = FastAPI(title="Pixels Mind API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directories for uploads and generated assets
STATIC_DIR = os.path.abspath("static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
GEN_DIR = os.path.join(STATIC_DIR, "generated")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GEN_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----- Schemas -----
class GenerateRequest(BaseModel):
    user_id: Optional[str] = None
    preset_id: Optional[str] = None
    aspect_ratio: Literal["portrait", "square", "landscape", "stories"] = "square"
    num_images: int = Field(1, ge=1, le=8)
    packs: Dict[str, Any] = Field(default_factory=dict)
    model_upload_ids: List[str] = Field(default_factory=list)
    apparel_upload_ids: List[str] = Field(default_factory=list)


# ----- Presets (static for now) -----
PRESETS = [
    {
        "id": "studio-soft",
        "title": "Studio Soft",
        "description": "Clean studio, softbox lighting, neutral gradient",
        "lighting": "softbox",
        "background": "studio gradient",
        "pose": "standing 3/4",
        "accent": "#9f74ff",
    },
    {
        "id": "editorial-dramatic",
        "title": "Editorial Dramatic",
        "description": "High contrast, rim lights, fashion mag style",
        "lighting": "dramatic",
        "background": "paper sweep",
        "pose": "dynamic",
        "accent": "#8b5cf6",
    },
    {
        "id": "lifestyle-sun",
        "title": "Lifestyle Sun",
        "description": "Golden hour outdoor patio, sun flares",
        "lighting": "golden hour",
        "background": "outdoor patio",
        "pose": "casual seated",
        "accent": "#a78bfa",
    },
    {
        "id": "god-mode",
        "title": "Creative God Mode",
        "description": "8 creative images with randomized styles",
        "lighting": "mixed",
        "background": "varied",
        "pose": "varied",
        "accent": "#9f74ff",
    },
]


# ----- Utils -----
from PIL import Image, ImageDraw, ImageFont  # pillow
from bson import ObjectId


def objid_str(oid: ObjectId | str) -> str:
    return str(oid)


def ar_to_size(ar: str) -> tuple[int, int]:
    if ar == "portrait":
        return (896, 1152)
    if ar == "landscape":
        return (1280, 832)
    if ar == "stories":
        return (896, 1600)
    return (1024, 1024)  # square


async def simulate_generation(job_id: str, req: GenerateRequest):
    """Simulate AI pipeline by creating placeholder images and updating job status in DB."""
    # Update job status -> processing
    db["job"].update_one({"_id": ObjectId(job_id)}, {"$set": {"status": "processing", "updated_at": datetime.utcnow()}})

    try:
        w, h = ar_to_size(req.aspect_ratio)
        num = req.num_images
        # Small delay to simulate stages
        stages = [
            "Image editing / model fitting",
            "Background swapping",
            "Pose transfer",
            "Apparel try-on",
            "Final image generation",
        ]
        for _ in stages:
            await asyncio.sleep(0.3)
        assets: list[str] = []
        for i in range(num):
            img = Image.new("RGB", (w, h), color=(8, 10, 16))
            draw = ImageDraw.Draw(img)
            # Accent bar
            draw.rectangle([(0, 0), (w, 8)], fill=(159, 116, 255))
            # Text
            txt = f"Pixels Mind\nJob {job_id[-6:]}\n{req.aspect_ratio.upper()} #{i+1}"
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((24, 24), txt, fill=(220, 220, 235), font=font, spacing=6)
            # Simple shape for vibe
            draw.rounded_rectangle([(w//2-160, h//2-160), (w//2+160, h//2+160)], radius=24, outline=(159,116,255), width=4)

            fname = f"gen_{job_id}_{i+1}.png"
            fpath = os.path.join(GEN_DIR, fname)
            img.save(fpath, format="PNG")
            url = f"/static/generated/{fname}"
            asset_id = create_document("asset", {"user_id": req.user_id, "job_id": job_id, "url": url, "kind": "generated", "meta": {"i": i}})
            assets.append(asset_id)
            await asyncio.sleep(0.05)

        db["job"].update_one({"_id": ObjectId(job_id)}, {"$set": {"status": "completed", "asset_ids": assets, "updated_at": datetime.utcnow()}})
    except Exception as e:
        db["job"].update_one({"_id": ObjectId(job_id)}, {"$set": {"status": "failed", "error": str(e), "updated_at": datetime.utcnow()}})


# ----- Routes -----
@app.get("/")
def root():
    return {"name": "Pixels Mind API", "status": "ok"}


@app.get("/api/presets")
def get_presets():
    return PRESETS


@app.get("/schema")
def get_schema():
    # Provide collection schemas to external viewers
    from schemas import User, Upload, Job, Asset, Preset
    return {
        "user": User.model_json_schema(),
        "upload": Upload.model_json_schema(),
        "job": Job.model_json_schema(),
        "asset": Asset.model_json_schema(),
        "preset": Preset.model_json_schema(),
    }


@app.post("/api/upload")
async def upload_files(
    kind: Literal["model", "apparel"] = Form(...),
    user_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    saved = []
    for file in files:
        # Basic validation
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Only PNG/JPG allowed")
        suffix = os.path.splitext(file.filename)[1].lower()
        fname = f"{datetime.utcnow().timestamp():.0f}_{os.urandom(3).hex()}{suffix}"
        dest = os.path.join(UPLOAD_DIR, fname)
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        url = f"/static/uploads/{fname}"
        up_id = create_document("upload", {"user_id": user_id, "kind": kind, "filename": file.filename, "url": url})
        saved.append({"id": up_id, "url": url, "filename": file.filename, "kind": kind})
    return {"count": len(saved), "items": saved}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    # Create job document
    job_id = create_document("job", {
        "user_id": req.user_id,
        "preset_id": req.preset_id,
        "aspect_ratio": req.aspect_ratio,
        "num_images": req.num_images,
        "packs": req.packs,
        "status": "queued",
        "asset_ids": [],
    })
    # Kick off async simulation
    asyncio.create_task(simulate_generation(job_id, req))
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    doc = db["job"].find_one({"_id": ObjectId(job_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    # Expand asset docs
    assets = []
    for aid in doc.get("asset_ids", []):
        a = db["asset"].find_one({"_id": ObjectId(aid)})
        if a:
            a["id"] = str(a.pop("_id"))
            assets.append(a)
    doc["id"] = str(doc.pop("_id"))
    doc["assets"] = assets
    return doc


@app.get("/api/jobs/{job_id}/zip")
async def job_zip(job_id: str):
    doc = db["job"].find_one({"_id": ObjectId(job_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    asset_docs = [db["asset"].find_one({"_id": ObjectId(aid)}) for aid in doc.get("asset_ids", [])]
    # Stream zip
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for a in asset_docs:
            if not a:
                continue
            url = a.get("url", "")
            if not url.startswith("/static/"):
                continue
            fpath = os.path.abspath(url.lstrip("/"))
            if os.path.exists(fpath):
                zf.write(fpath, arcname=os.path.basename(fpath))
    mem.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=pixels-mind-{job_id[-6:]}.zip"}
    return StreamingResponse(mem, headers=headers, media_type="application/zip")


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
