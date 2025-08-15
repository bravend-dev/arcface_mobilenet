# main.py
# -----------------------------
# FastAPI Face Search Demo (single-file)
# - Serves a minimal but clean HTML UI
# - Scans ./media for gallery images at startup
# - Provides POST /api/search to return top-k nearest images + labels
# - Serves images from /media/<path>
#
# To run:
#   1) Put some images under ./media (optionally grouped in subfolders).
#   2) (Optional) Create encoder.py with a function `encode(pil_image) -> 1D np.ndarray`.
#      If not present, a simple color-histogram fallback encoder is used.
#   3) pip install fastapi uvicorn[standard] numpy pillow python-multipart
#   4) python main.py
#   5) Open http://127.0.0.1:8000

from __future__ import annotations
import os
import io
import sys
import json
import pathlib
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# -----------------------------
# Optional: Import user-provided encoder from encoder.py
# Must define: encode(pil_image: PIL.Image.Image) -> np.ndarray (1D feature)
# If absent, we'll use a simple color histogram fallback so the demo still works.
# -----------------------------
from mb_encoder import Encoder, crop_largest_face
encoder = Encoder(device="cpu")

# -----------------------------
# Config
# -----------------------------
MEDIA_DIR = pathlib.Path("media").resolve()
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TOPK_DEFAULT = 5
MAX_TOPK = 30

app = FastAPI(title="Face Search Demo", version="0.1.0")

# CORS (front-end is same origin, but this keeps things flexible)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static /media for direct image serving
if MEDIA_DIR.exists():
    app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
else:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


# --- Ví dụ tích hợp với encoder của bạn ---
# giả sử bạn có biến `encoder` với hàm encode_img(pil_image)
def encode_with_face_crop(pil_img: Image.Image, encoder, **encode_kwargs):
    face_img = crop_largest_face(pil_img)
    return encoder.encode_img(face_img, **encode_kwargs)

# -----------------------------
# Cosine similarity Top-K retrieval (based on user's evaluate_search logic)
# -----------------------------

def cosine_topk(q_feats: np.ndarray, g_feats: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k gallery for each query row.
    q_feats: (Q, D)  g_feats: (G, D)
    """
    q_norm = np.linalg.norm(q_feats, axis=1, keepdims=True)
    g_norm = np.linalg.norm(g_feats, axis=1, keepdims=True).T  # (1, G)
    sims = (q_feats @ g_feats.T) / (q_norm * g_norm + 1e-6)
    # Argpartition for efficiency, then sort top-k
    k = min(k, g_feats.shape[0])
    part = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    # sort each row's top-k
    rows = np.arange(sims.shape[0])[:, None]
    sorted_idx = part[rows, np.argsort(-sims[rows, part])]
    sorted_scores = sims[rows, sorted_idx]
    return sorted_idx, sorted_scores

# -----------------------------
# Gallery indexing (scan media folder -> features, labels, relpaths)
# -----------------------------
G_LABELS: List[str] = []
G_PATHS: List[str] = []  # relative URL paths under /media
G_FEATS: np.ndarray | None = None


def infer_label_from_path(path: pathlib.Path) -> str:
    # Prefer parent folder name as label, else use stem
    if path.parent != MEDIA_DIR:
        return path.parent.name
    return path.stem


def iter_gallery_images(root: pathlib.Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            yield p


def build_gallery_index():
    global G_LABELS, G_PATHS, G_FEATS
    labels: List[str] = []
    paths: List[str] = []
    feats: List[np.ndarray] = []

    files = list(iter_gallery_images(MEDIA_DIR))
    if not files:
        print(f"[warn] No images found in {MEDIA_DIR}. Place images to enable search.")

    feats = encoder.encode_paths(files)
    for i, fpath in enumerate(sorted(files)):
        label = infer_label_from_path(fpath)
        rel = str(fpath.relative_to(MEDIA_DIR)).replace(os.sep, "/")
        labels.append(label)
        paths.append(rel)
        if (i + 1) % 20 == 0:
            print(f"Indexed {i+1} images...")

    G_FEATS = feats.astype(np.float32)
    G_LABELS = labels
    G_PATHS = paths


build_gallery_index()

# -----------------------------
# Routes
# -----------------------------

@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(content=_INDEX_HTML, status_code=200)


@app.post("/api/search")
async def api_search(
    file: UploadFile = File(...),
    topk: int = Query(TOPK_DEFAULT, ge=1, le=MAX_TOPK),
):
    if G_FEATS is None or G_FEATS.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Gallery is empty. Add images under ./media and restart.")

    try:
        content = await file.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image upload.")

    # q_feat = encoder.encode_img(pil)
    q_feat = encode_with_face_crop(pil, encoder=encoder)
    idxs, scores = cosine_topk(q_feat, G_FEATS, k=topk)

    results = []
    for rank, (gi, sc) in enumerate(zip(idxs[0].tolist(), scores[0].tolist()), start=1):
        results.append({
            "rank": rank,
            "label": G_LABELS[gi],
            "score": float(sc),
            "image_url": f"/media/{G_PATHS[gi]}",
            "path": G_PATHS[gi],  # relative path under media
        })

    return JSONResponse({
        "count": len(results),
        "topk": topk,
        "results": results,
    })


@app.get("/api/media/list")
def api_media_list():
    items = [
        {"label": G_LABELS[i], "image_url": f"/media/{G_PATHS[i]}", "path": G_PATHS[i]}
        for i in range(len(G_PATHS))
    ]
    return JSONResponse({"count": len(items), "items": items})


@app.get("/api/health")
def api_health():
    return {"status": "ok", "gallery_size": len(G_PATHS)}


# Optional direct file API (StaticFiles already serves /media/*)
@app.get("/api/media/{rel_path:path}")
def api_media_get(rel_path: str):
    fs_path = MEDIA_DIR / rel_path
    if not fs_path.exists() or not fs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(fs_path))


# -----------------------------
# Frontend (HTML/CSS/JS)
# -----------------------------
_INDEX_HTML = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Search Demo</title>
  <style>
    :root {
      --bg: #0b0f19;
      --panel: #12182a;
      --soft: #1a2136;
      --text: #e6e9f2;
      --muted: #9aa4c7;
      --brand: #6ea8fe;
      --ring: rgba(110,168,254,0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, "Helvetica Neue", Arial;
      background: linear-gradient(180deg, var(--bg), #0c1326 60%, #0b1023);
      color: var(--text);
    }
    .container { max-width: 980px; margin: 40px auto; padding: 0 16px; }
    .header { display:flex; align-items:center; gap:12px; }
    .logo {
      width: 40px; height: 40px; border-radius: 12px; background: radial-gradient(circle at 30% 30%, #6ea8fe, #8e6efe 60%, #3a60ff);
      box-shadow: 0 10px 30px rgba(110,168,254,0.35);
    }
    h1 { font-size: 28px; margin: 0; letter-spacing: 0.2px; }
    p.muted { color: var(--muted); margin: 4px 0 0; }

    .card {
      background: linear-gradient(180deg, var(--panel), var(--soft));
      border: 1px solid #1d2741; border-radius: 20px; padding: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }

    .uploader { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: stretch; }
    .dropzone {
      border: 2px dashed #2a3556; border-radius: 16px; padding: 16px; min-height: 210px;
      display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; text-align:center;
      background: rgba(255,255,255,0.02);
      transition: border-color .2s, box-shadow .2s, transform .2s;
    }
    .dropzone.drag {
      border-color: var(--brand); box-shadow: 0 0 0 6px var(--ring); transform: translateY(-2px);
    }
    .dz-icon { font-size: 40px; opacity: .9; }
    .dz-title { font-weight: 700; }
    .dz-note { color: var(--muted); font-size: 14px; }

    .controls { display:flex; gap:12px; align-items:center; }
    .select {
      background:#0f1628; color:var(--text); border:1px solid #273256; border-radius:12px;
      padding:10px 12px; outline:none;
    }
    .btn {
      appearance:none; border:none; border-radius:14px; padding:10px 16px; font-weight:700;
      background: linear-gradient(180deg, #7eb0ff, #5a83ff); color:#0a1020; cursor:pointer;
      box-shadow: 0 10px 30px rgba(110,168,254,0.35), inset 0 0 0 1px rgba(255,255,255,.2);
      transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
    }
    .btn:disabled { opacity:.5; cursor:not-allowed; filter:grayscale(40%); }
    .btn:hover:not(:disabled) { transform: translateY(-1px); }

    .preview { border-radius: 14px; overflow: hidden; background:#0f1527; border:1px solid #273256; display:flex; align-items:center; justify-content:center; }
    .preview img { max-width:100%; max-height: 210px; display:block; }

    .results { margin-top: 18px; display:grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr)); gap: 14px; }
    .item { background:#0f1527; border:1px solid #273256; border-radius: 14px; overflow:hidden; }
    .thumb { aspect-ratio: 1 / 1; width: 100%; object-fit: cover; display:block; }
    .meta { padding: 10px 12px; display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .lbl { font-weight:700; font-size: 14px; white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }
    .score { color: var(--muted); font-size: 12px; }

    .footer { margin-top: 18px; color: var(--muted); font-size: 12px; text-align:center; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo" aria-hidden="true"></div>
      <div>
        <h1>Face Search Demo</h1>
        <p class="muted">Upload 1 ảnh, nhận về danh sách top‑k ảnh gần nhất (cosine similarity).</p>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <div class="uploader">
        <div class="dropzone" id="dropzone">
          <div class="dz-icon">📷</div>
          <div class="dz-title">Kéo thả ảnh vào đây hoặc chọn file</div>
          <div class="dz-note">Hỗ trợ: JPG, PNG, WEBP, BMP</div>
          <input id="file" type="file" accept="image/*" style="margin-top:10px;" />
        </div>
        <div class="preview" id="preview"><span style="color:#7481ab;">(Xem trước ảnh)</span></div>
      </div>
      <div style="height:12px;"></div>
      <div class="controls">
        <label for="topk">Top‑k:</label>
        <select id="topk" class="select">
          <option value="1">1</option>
          <option value="3">3</option>
          <option value="5" selected>5</option>
          <option value="10">10</option>
          <option value="20">20</option>
        </select>
        <button id="btnSearch" class="btn" disabled>Tìm kiếm</button>
        <span id="status" class="muted"></span>
      </div>
    </div>

    <div class="results" id="results"></div>
    <div class="footer">Backend: FastAPI • Similarity: Cosine • UI: HTML/CSS/JS thuần</div>
  </div>

  <script>
    const dz = document.getElementById('dropzone');
    const fileInput = document.getElementById('file');
    const btnSearch = document.getElementById('btnSearch');
    const preview = document.getElementById('preview');
    const results = document.getElementById('results');
    const topkSel = document.getElementById('topk');
    const statusEl = document.getElementById('status');

    let chosenFile = null;

    function setStatus(msg) { statusEl.textContent = msg || ''; }

    function showPreview(file) {
      preview.innerHTML = '';
      const img = document.createElement('img');
      img.alt = 'preview';
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);
      preview.appendChild(img);
    }

    function enableBtn() { btnSearch.disabled = !chosenFile; }

    fileInput.addEventListener('change', (e) => {
      chosenFile = e.target.files?.[0] || null;
      if (chosenFile) showPreview(chosenFile);
      enableBtn();
    });

    ;['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, (e)=>{
      e.preventDefault(); e.stopPropagation(); dz.classList.add('drag');
    }));
    ;['dragleave','drop'].forEach(ev => dz.addEventListener(ev, (e)=>{
      e.preventDefault(); e.stopPropagation(); dz.classList.remove('drag');
    }));
    dz.addEventListener('drop', (e) => {
      const f = e.dataTransfer?.files?.[0];
      if (f) { chosenFile = f; fileInput.files = e.dataTransfer.files; showPreview(f); enableBtn(); }
    });

    btnSearch.addEventListener('click', async () => {
      if (!chosenFile) return;
      setStatus('Đang tìm...');
      results.innerHTML = '';

      const fd = new FormData();
      fd.append('file', chosenFile);
      const topk = topkSel.value;

      try {
        const res = await fetch(`/api/search?topk=${topk}`, { method: 'POST', body: fd });
        if (!res.ok) {
          const err = await res.json().catch(()=>({detail:'Error'}));
          throw new Error(err.detail || res.statusText);
        }
        const data = await res.json();
        setStatus(`Tìm thấy ${data.count} kết quả.`);
        renderResults(data.results || []);
      } catch (err) {
        console.error(err);
        setStatus('Lỗi: ' + (err.message || 'Không xác định'));
      }
    });

    function renderResults(items) {
      results.innerHTML = '';
      for (const it of items) {
        const card = document.createElement('div');
        card.className = 'item';
        const img = document.createElement('img');
        img.className = 'thumb';
        img.src = it.image_url;
        img.alt = it.label;
        const meta = document.createElement('div');
        meta.className = 'meta';
        const lbl = document.createElement('div');
        lbl.className = 'lbl';
        lbl.textContent = `#${it.rank} · ${it.label}`;
        const score = document.createElement('div');
        score.className = 'score';
        score.textContent = (it.score*100).toFixed(1) + '%';
        meta.appendChild(lbl);
        meta.appendChild(score);
        card.appendChild(img);
        card.appendChild(meta);
        results.appendChild(card);
      }
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
