import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.models import Patient, VariantInput, StructureRequest
from backend.services import (
    StructureService, VariantAnalysisService, 
    ACMGClassifier, ReportGenerator
)
from backend.database import get_db, init_db
from backend.auth import get_current_user
from backend.cache import CacheManager

# Configure logging for cloud deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
structure_service = StructureService()
variant_service = VariantAnalysisService()
acmg_classifier = ACMGClassifier()
report_generator = ReportGenerator()
cache_manager = CacheManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Clinical Genome Visualizer...")
    await init_db()
    await cache_manager.connect()
    yield
    # Shutdown
    logger.info("Shutting down...")
    await cache_manager.disconnect()

# Create FastAPI app
app = FastAPI(
    title="Clinical Genome Visualizer",
    description="3D visualization and analysis of genetic variants",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files - create directory if it doesn't exist
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static directory not found, skipping static file mounting")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Patient endpoints
@app.post("/api/patients")
async def create_patient(patient: Patient, db=Depends(get_db)):
    """Create new patient record"""
    try:
        result = await db.execute(
            "INSERT INTO patients (patient_id, name, birth_date, sex) VALUES (?, ?, ?, ?)",
            (patient.patient_id, patient.name, patient.birth_date, patient.sex)
        )
        await db.commit()
        return {"message": "Patient created", "id": result.lastrowid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str, db=Depends(get_db)):
    """Get patient by ID"""
    cursor = await db.execute(
        "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
    )
    patient = await cursor.fetchone()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return dict(patient)

# Variant analysis
@app.post("/api/variants/analyze")
async def analyze_variant(variant: VariantInput, background_tasks: BackgroundTasks):
    """Analyze genetic variant"""
    try:
        cache_key = f"variant:{variant.gene}:{variant.protein_change}"
        cached_result = await cache_manager.get(cache_key)
        
        if cached_result:
            return cached_result
        
        classification = await acmg_classifier.classify(variant)
        structure_impact = await variant_service.predict_structure_impact(variant)
        
        result = {
            "variant": variant.model_dump(),
            "classification": classification,
            "structure_impact": structure_impact,
            "timestamp": datetime.now().isoformat()
        }
        
        await cache_manager.set(cache_key, result, ttl=3600)
        return result
        
    except Exception as e:
        logger.error(f"Variant analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Structure visualization
@app.post("/api/structure/visualize")
async def get_structure(request: StructureRequest):
    """Get 3D structure data"""
    try:
        pdb_data = await structure_service.fetch_pdb(request.gene)
        if not pdb_data:
            raise HTTPException(status_code=404, detail=f"No structure found for {request.gene}")
        
        if request.variant:
            pdb_data = await structure_service.annotate_variant(pdb_data, request.variant)
        
        return JSONResponse(content=pdb_data)
        
    except Exception as e:
        logger.error(f"Structure visualization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload
@app.post("/api/upload/vcf")
async def upload_vcf(file: UploadFile = File(...), patient_id: str = None):
    """Upload VCF file"""
    try:
        if not file.filename.endswith(('.vcf', '.vcf.gz')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        file_path = Path(f"./uploads/{patient_id}/{file.filename}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Report generation
@app.post("/api/report/generate")
async def generate_report(patient_id: str, format: str = "json"):
    """Generate clinical report"""
    try:
        if format == "json":
            report_data = await report_generator.generate_json(patient_id, [])
            return JSONResponse(content=report_data)
        else:
            raise HTTPException(status_code=400, detail="Only JSON format supported")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({
                "patient_id": patient_id,
                "message": data,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )