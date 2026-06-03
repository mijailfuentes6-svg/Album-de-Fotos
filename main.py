from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import shutil
import io
import zipfile
from typing import List

# Importamos la lógica del motor de IA
from ml_engine import cluster_images
from train_model import train

app = FastAPI(title="Álbum de Fotos API", description="Motor de clustering no supervisado")

# Configuración del Sistema de Archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = "/data/pictures" 

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ScanRequest(BaseModel):
    num_clusters: int = 3

# ==========================================
# ENDPOINT: FRONTEND (RAÍZ)
# ==========================================
@app.get("/")
async def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Archivo index.html no encontrado en la carpeta static"}

# ==========================================
# ENDPOINTS DE LA API (Fases 1, 2 y 3)
# ==========================================
@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    try:
        # Limpiamos el directorio antes de un nuevo entrenamiento
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        saved_count = 0
        for file in files:
            if file.content_type.startswith("image/"):
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_count += 1
                
        return {"status": "success", "message": f"Dataset cargado: {saved_count} imágenes listas para el entrenamiento."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en I/O de disco: {str(e)}")

@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(train)
        return {"status": "success", "message": "Proceso de optimización neuronal iniciado."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan")
async def scan_directory(request: ScanRequest):
    if not os.listdir(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="El dataset está vacío. Sube imágenes primero.")
    try:
        resultado = cluster_images(UPLOAD_DIR, request.num_clusters)
        return {"status": "success", "clusters": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image")
async def get_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(path)

# ==========================================
# ENDPOINT: DESCARGA DE ÁLBUMES (ZIP)
# ==========================================
@app.post("/api/download-zip")
async def download_albums_zip(albums: dict = Body(...)):
    """
    Recibe el JSON estructurado de los clústeres desde el frontend y empaqueta
    las imágenes en un archivo ZIP estructurado por carpetas en la memoria RAM.
    """
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for album_name, photos in albums.items():
                for photo_item in photos:
                    # Extraemos el nombre base por si el frontend envía la ruta completa
                    filename = os.path.basename(photo_item)
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    
                    if os.path.exists(file_path):
                        # arcname define la estructura interna dentro del ZIP
                        zip_file.write(file_path, arcname=f"{album_name}/{filename}")
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer, 
            media_type="application/zip", 
            headers={"Content-Disposition": "attachment; filename=albumes_agrupados.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando el archivo ZIP: {str(e)}")