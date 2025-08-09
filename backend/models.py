"""
Data Models and Schemas for Clinical Genome Visualizer
SQLAlchemy ORM Models and Pydantic Schemas
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, validator
from bson import ObjectId

# SQLAlchemy Base
Base = declarative_base()

# ==================== Enums ====================

class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"

class ClinicalSignificance(str, Enum):
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    VUS = "VUS"  # Variant of Uncertain Significance
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"
    
class Zygosity(str, Enum):
    HOMOZYGOUS = "Homozygous"
    HETEROZYGOUS = "Heterozygous"
    HEMIZYGOUS = "Hemizygous"
    
class VariantType(str, Enum):
    SNV = "SNV"  # Single Nucleotide Variant
    INDEL = "Indel"
    CNV = "CNV"  # Copy Number Variant
    SV = "SV"  # Structural Variant

# ==================== SQLAlchemy Models ====================

class PatientModel(Base):
    """Patient database model"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    birth_date = Column(DateTime, nullable=False)
    sex = Column(String(1), nullable=False)
    
    # Clinical information
    clinical_diagnosis = Column(Text)
    family_history = Column(JSON)
    medications = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    
    # PHI encryption flag
    is_encrypted = Column(Boolean, default=False)
    
    # Relationships
    variants = relationship("VariantModel", back_populates="patient", cascade="all, delete-orphan")
    reports = relationship("ReportModel", back_populates="patient", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Patient(patient_id={self.patient_id}, name={self.name})>"

class VariantModel(Base):
    """Genetic variant database model"""
    __tablename__ = "variants"
    
    id = Column(Integer, primary_key=True, index=True)
    variant_id = Column(String(100), unique=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"))
    
    # Genomic information
    gene = Column(String(50), index=True, nullable=False)
    chromosome = Column(String(10), nullable=False)
    position = Column(Integer, nullable=False)
    ref = Column(String(1000))  # Reference allele
    alt = Column(String(1000))  # Alternative allele
    
    # Variant details
    variant_type = Column(String(20))
    transcript = Column(String(50))
    protein_change = Column(String(100))
    cdna_change = Column(String(100))
    
    # Metrics
    vaf = Column(Float)  # Variant Allele Frequency
    depth = Column(Integer)  # Read depth
    quality = Column(Float)  # Quality score
    zygosity = Column(String(20))
    
    # Classification
    classification = Column(String(50))
    acmg_criteria = Column(JSON)
    confidence_score = Column(Float)
    
    # Clinical annotations
    clinvar_id = Column(String(50))
    cosmic_id = Column(String(50))
    dbsnp_id = Column(String(50))
    gnomad_af = Column(Float)  # Population frequency
    
    # Structure impact
    structure_impact = Column(JSON)
    drug_interactions = Column(JSON)
    
    # Metadata
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analyzed_by = Column(String(100))
    
    # Relationships
    patient = relationship("PatientModel", back_populates="variants")
    
    def __repr__(self):
        return f"<Variant(gene={self.gene}, change={self.protein_change})>"

class ReportModel(Base):
    """Clinical report database model"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(100), unique=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"))
    
    # Report details
    report_type = Column(String(50))  # comprehensive, targeted, pharmacogenomics
    status = Column(String(20))  # draft, final, amended
    
    # Content
    summary = Column(Text)
    findings = Column(JSON)
    recommendations = Column(JSON)
    
    # Files
    pdf_path = Column(String(500))
    json_data = Column(JSON)
    
    # Metadata
    generated_at = Column(DateTime, default=datetime.utcnow)
    generated_by = Column(String(100))
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    
    # Relationships
    patient = relationship("PatientModel", back_populates="reports")

# ==================== Pydantic Schemas ====================

class PatientBase(BaseModel):
    """Base patient schema"""
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Patient name")
    birth_date: datetime = Field(..., description="Date of birth")
    sex: Sex = Field(..., description="Biological sex")
    
class PatientCreate(PatientBase):
    """Schema for creating patient"""
    clinical_diagnosis: Optional[str] = None
    family_history: Optional[Dict[str, Any]] = None
    medications: Optional[List[str]] = None
    
class Patient(PatientBase):
    """Complete patient schema"""
    id: int
    clinical_diagnosis: Optional[str]
    family_history: Optional[Dict[str, Any]]
    medications: Optional[List[str]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class VariantBase(BaseModel):
    """Base variant schema"""
    gene: str = Field(..., description="Gene symbol")
    chromosome: str = Field(..., description="Chromosome")
    position: int = Field(..., description="Genomic position")
    ref: str = Field(..., description="Reference allele")
    alt: str = Field(..., description="Alternative allele")
    
    @validator('gene')
    def validate_gene(cls, v):
        """Validate gene symbol"""
        if not v or not v.isupper():
            raise ValueError('Gene symbol must be uppercase')
        return v
    
class VariantCreate(VariantBase):
    """Schema for creating variant"""
    transcript: Optional[str] = None
    protein_change: Optional[str] = None
    cdna_change: Optional[str] = None
    vaf: float = Field(0.5, ge=0, le=1)
    depth: Optional[int] = Field(None, ge=0)
    quality: Optional[float] = Field(None, ge=0)
    zygosity: Zygosity = Zygosity.HETEROZYGOUS
    
class Variant(VariantBase):
    """Complete variant schema"""
    id: int
    variant_id: str
    patient_id: str
    variant_type: Optional[str]
    transcript: Optional[str]
    protein_change: Optional[str]
    classification: Optional[ClinicalSignificance]
    acmg_criteria: Optional[List[str]]
    confidence_score: Optional[float]
    structure_impact: Optional[Dict[str, Any]]
    analyzed_at: datetime
    
    class Config:
        from_attributes = True

class VariantAnalysisResult(BaseModel):
    """Variant analysis result schema"""
    variant: Variant
    classification: ClinicalSignificance
    acmg_criteria: List[str]
    confidence_score: float = Field(..., ge=0, le=1)
    structure_impact: Dict[str, Any]
    clinical_annotations: Dict[str, Any]
    recommendations: List[str]
    
class StructureData(BaseModel):
    """3D structure data schema"""
    pdb_id: str
    gene: str
    atoms: List[List[float]]  # [[x, y, z], ...]
    colors: Optional[List[List[float]]]  # [[r, g, b], ...]
    chains: Optional[Dict[str, Any]]
    variants: Optional[List[Dict[str, Any]]]
    density_map: Optional[str]  # URL or base64
    alphafold_confidence: Optional[List[float]]
    
class ClinicalReport(BaseModel):
    """Clinical report schema"""
    report_id: str
    patient: Patient
    variants: List[Variant]
    summary: str
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime
    generated_by: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DrugInteraction(BaseModel):
    """Drug-gene interaction schema"""
    gene: str
    drug: str
    interaction_type: str  # inhibitor, substrate, inducer
    evidence_level: str  # strong, moderate, weak
    clinical_significance: str
    recommendation: str
    references: List[str]
    
class ACMGCriteria(BaseModel):
    """ACMG classification criteria"""
    criterion: str  # PVS1, PS1, PM1, etc.
    category: str  # pathogenic_very_strong, pathogenic_strong, etc.
    met: bool
    evidence: str
    
class AnalysisRequest(BaseModel):
    """Analysis request schema"""
    patient_id: str
    variants: List[VariantCreate]
    analysis_type: str = "comprehensive"  # comprehensive, targeted, rapid
    include_alphafold: bool = True
    include_drug_interactions: bool = True
    generate_report: bool = True
    
class AnalysisResponse(BaseModel):
    """Analysis response schema"""
    request_id: str
    status: str  # pending, processing, completed, failed
    results: Optional[List[VariantAnalysisResult]]
    report_url: Optional[str]
    completed_at: Optional[datetime]
    error: Optional[str]

# ==================== MongoDB Models (Optional) ====================

class MongoBaseModel(BaseModel):
    """Base model for MongoDB documents"""
    id: Optional[str] = Field(None, alias="_id")
    
    @validator("id", pre=True)
    def validate_id(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        return v
    
    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }

class AnalysisJob(MongoBaseModel):
    """Analysis job for task queue"""
    job_id: str
    patient_id: str
    variants: List[Dict[str, Any]]
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    
class AuditLog(MongoBaseModel):
    """HIPAA audit log"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    action: str  # view, create, update, delete, export
    resource_type: str  # patient, variant, report
    resource_id: str
    ip_address: str
    user_agent: str
    success: bool
    details: Optional[Dict[str, Any]]