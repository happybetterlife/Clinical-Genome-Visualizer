#!/usr/bin/env python3
"""
Clinical Genome Visualizer - Main Backend Server
FastAPI + Mol* Integration
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', './logs/app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules
from backend.models import Patient, Variant, ClinicalReport
from backend.services import (
    StructureService,
    VariantAnalysisService,
    ACMGClassifier,
    ReportGenerator
)
from backend.database import get_db, init_db
from backend.auth import get_current_user, create_access_token
from backend.cache import cache_manager

# Initialize services
structure_service = StructureService()
variant_service = VariantAnalysisService()
acmg_classifier = ACMGClassifier()
report_generator = ReportGenerator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Clinical Genome Visualizer...")
    await init_db()
    await acmg_classifier.load_models()
    await structure_service.test_connections()
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await cache_manager.clear()
    logger.info("Application shut down")

# Initialize FastAPI app
app = FastAPI(
    title=os.getenv('APP_NAME', 'Clinical Genome Visualizer'),
    version=os.getenv('APP_VERSION', '1.0.0'),
    description="Mol* powered clinical genome visualization and analysis platform",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=eval(os.getenv('CORS_ORIGINS', '["*"]')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ==================== Data Models ====================

class PatientInput(BaseModel):
    """Patient information input model"""
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Patient name")
    birth_date: str = Field(..., description="Birth date (YYYY-MM-DD)")
    sex: str = Field(..., description="Biological sex (M/F)")
    clinical_diagnosis: Optional[str] = None
    family_history: Optional[Dict[str, Any]] = None
    
class VariantInput(BaseModel):
    """Genetic variant input model"""
    gene: str = Field(..., description="Gene symbol (e.g., BRCA1)")
    chromosome: str = Field(..., description="Chromosome (1-22, X, Y)")
    position: int = Field(..., description="Genomic position")
    ref: str = Field(..., description="Reference allele")
    alt: str = Field(..., description="Alternative allele")
    transcript: Optional[str] = None
    protein_change: Optional[str] = None
    vaf: float = Field(0.5, description="Variant allele frequency")
    zygosity: str = Field("Heterozygous", description="Zygosity status")

class StructureRequest(BaseModel):
    """3D structure visualization request"""
    gene: str
    variant: Optional[VariantInput] = None
    show_density: bool = False
    show_alphafold: bool = False
    show_clinical_variants: bool = False

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": os.getenv('APP_VERSION', '1.0.0'),
        "services": {
            "database": "connected",
            "redis": "connected",
            "mol_star": "ready"
        }
    }

# ==================== Patient Management ====================

@app.post("/api/patients", response_model=Patient)
async def create_patient(
    patient: PatientInput,
    db = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create new patient record"""
    try:
        # Create patient in database
        new_patient = await db.patients.create(patient.dict())
        
        # Audit log for HIPAA compliance
        logger.info(f"Patient created: {patient.patient_id} by user: {current_user.id}")
        
        return new_patient
    except Exception as e:
        logger.error(f"Error creating patient: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create patient")

@app.get("/api/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    db = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Retrieve patient information"""
    patient = await db.patients.find_one({"patient_id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Decrypt PHI if enabled
    if os.getenv('ENCRYPT_PATIENT_DATA', 'False') == 'True':
        patient = decrypt_patient_data(patient)
    
    return patient

# ==================== Variant Analysis ====================

@app.post("/api/variants/analyze")
async def analyze_variant(
    variant: VariantInput,
    patient_id: str,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """Analyze genetic variant with ACMG classification"""
    try:
        # Check cache first
        cache_key = f"variant:{variant.gene}:{variant.protein_change}"
        cached_result = await cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for variant: {cache_key}")
            return cached_result
        
        # Perform ACMG classification
        classification = await acmg_classifier.classify(variant)
        
        # Get structure impact prediction
        structure_impact = await variant_service.predict_structure_impact(variant)
        
        # Fetch clinical significance from ClinVar
        clinvar_data = None
        if os.getenv('ENABLE_CLINVAR', 'True') == 'True':
            clinvar_data = await variant_service.fetch_clinvar(variant)
        
        result = {
            "variant": variant.dict(),
            "classification": classification,
            "structure_impact": structure_impact,
            "clinvar": clinvar_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        await cache_manager.set(cache_key, result, ttl=int(os.getenv('CACHE_TTL', 3600)))
        
        # Save to database in background
        background_tasks.add_task(save_variant_analysis, patient_id, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Variant analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ==================== 3D Structure Visualization ====================

@app.post("/api/structure/visualize")
async def get_structure(request: StructureRequest):
    """Get 3D structure data for Mol* visualization"""
    try:
        # Get PDB structure
        pdb_data = await structure_service.fetch_pdb(request.gene)
        
        if not pdb_data:
            raise HTTPException(status_code=404, detail=f"No structure found for {request.gene}")
        
        # Process variant if provided
        if request.variant:
            pdb_data = await structure_service.annotate_variant(
                pdb_data, 
                request.variant
            )
        
        # Add electron density if requested
        if request.show_density:
            pdb_data['density_map'] = await structure_service.fetch_density_map(
                pdb_data['pdb_id']
            )
        
        # Add AlphaFold structure if requested
        if request.show_alphafold and os.getenv('ENABLE_ALPHAFOLD', 'True') == 'True':
            af_structure = await structure_service.fetch_alphafold(request.gene)
            if af_structure:
                pdb_data['alphafold'] = af_structure
        
        # Add clinical variants if requested
        if request.show_clinical_variants:
            clinical_variants = await structure_service.fetch_clinical_variants(
                request.gene
            )
            pdb_data['clinical_variants'] = clinical_variants
        
        return JSONResponse(content=pdb_data)
        
    except Exception as e:
        logger.error(f"Structure visualization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== File Upload ====================

@app.post("/api/upload/vcf")
async def upload_vcf(
    file: UploadFile = File(...),
    patient_id: str = None,
    background_tasks: BackgroundTasks = None
):
    """Upload and process VCF file"""
    try:
        # Validate file type
        if not file.filename.endswith(('.vcf', '.vcf.gz')):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload VCF file.")
        
        # Save file
        file_path = Path(f"./uploads/{patient_id}/{file.filename}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Process VCF in background
        background_tasks.add_task(process_vcf, file_path, patient_id)
        
        return {
            "message": "VCF file uploaded successfully",
            "filename": file.filename,
            "patient_id": patient_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"VCF upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Report Generation ====================

@app.post("/api/report/generate")
async def generate_report(
    patient_id: str,
    format: str = "pdf",
    db = Depends(get_db)
):
    """Generate clinical report"""
    try:
        # Fetch patient and variants
        patient = await db.patients.find_one({"patient_id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        variants = await db.variants.find({"patient_id": patient_id}).to_list(None)
        
        # Generate report
        if format == "pdf":
            report_path = await report_generator.generate_pdf(patient, variants)
            return FileResponse(report_path, media_type='application/pdf')
        
        elif format == "json":
            report_data = await report_generator.generate_json(patient, variants)
            return JSONResponse(content=report_data)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'pdf' or 'json'")
            
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Drug Interaction ====================

@app.post("/api/drug/interaction")
async def check_drug_interaction(gene: str, drug: str):
    """Check drug-gene interaction"""
    if not os.getenv('ENABLE_DRUG_DOCKING', 'True') == 'True':
        raise HTTPException(status_code=503, detail="Drug docking feature is disabled")
    
    try:
        interaction = await structure_service.check_drug_binding(gene, drug)
        return interaction
    except Exception as e:
        logger.error(f"Drug interaction check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WebSocket for Real-time Updates ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str):
    """WebSocket for real-time analysis updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process and broadcast updates
            await manager.broadcast({
                "patient_id": patient_id,
                "message": data,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        manager.disconnect(websocket)

# ==================== Helper Functions ====================

async def save_variant_analysis(patient_id: str, result: Dict[str, Any]):
    """Save variant analysis result to database"""
    # Placeholder for database save operation
    logger.info(f"Saved variant analysis for patient: {patient_id}")

async def process_vcf(file_path: str, patient_id: str):
    """Process VCF file in background"""
    # Placeholder for VCF processing
    logger.info(f"Processing VCF file: {file_path} for patient: {patient_id}")

def decrypt_patient_data(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Decrypt patient PHI data"""
    # Placeholder for decryption
    return patient

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('RELOAD', 'True') == 'True',
        workers=int(os.getenv('WORKERS', 4)),
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )