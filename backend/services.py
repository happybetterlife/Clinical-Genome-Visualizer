"""
Core Business Logic Services for Clinical Genome Visualizer
Structure handling, variant analysis, ACMG classification, report generation
"""

import os
import json
import logging
import hashlib
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import aiohttp
from Bio import PDB, SeqIO
from Bio.PDB import PDBParser, DSSP
from Bio.Seq import Seq
# Optional dependencies - will be imported when needed
# import mdtraj as md
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import torch
# import joblib

from backend.models import (
    Variant, VariantAnalysisResult, ClinicalSignificance,
    StructureData, DrugInteraction, ACMGCriteria
)
from backend.cache import cache_manager
from backend.ml_pipeline import VariantPredictor, StructureImpactPredictor

logger = logging.getLogger(__name__)

# ==================== Structure Service ====================

class StructureService:
    """Service for handling protein structures"""
    
    # Gene to PDB and UniProt mapping
    GENE_MAPPING = {
        'BRCA1': {'pdb': '1T15', 'uniprot': 'P38398', 'length': 1863},
        'BRCA2': {'pdb': '1MIU', 'uniprot': 'P51587', 'length': 3418},
        'TP53': {'pdb': '1TUP', 'uniprot': 'P04637', 'length': 393},
        'EGFR': {'pdb': '2ITY', 'uniprot': 'P00533', 'length': 1210},
        'KRAS': {'pdb': '4OBE', 'uniprot': 'P01116', 'length': 189},
        'PIK3CA': {'pdb': '4JPS', 'uniprot': 'P42336', 'length': 1068},
        'PTEN': {'pdb': '1D5R', 'uniprot': 'P60484', 'length': 403},
        'MLH1': {'pdb': '4P7A', 'uniprot': 'P40692', 'length': 756},
    }
    
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.base_url = os.getenv('RCSB_PDB_API', 'https://files.rcsb.org')
        self.alphafold_url = os.getenv('ALPHAFOLD_API', 'https://alphafold.ebi.ac.uk')
        
    async def fetch_pdb(self, gene: str) -> Optional[Dict[str, Any]]:
        """Fetch PDB structure for a gene"""
        
        # Check cache first
        cache_key = f"pdb:{gene}"
        cached = await cache_manager.get(cache_key)
        if cached:
            logger.info(f"Cache hit for PDB structure: {gene}")
            return cached
        
        mapping = self.GENE_MAPPING.get(gene)
        if not mapping:
            logger.error(f"No PDB mapping for gene: {gene}")
            return None
        
        pdb_id = mapping['pdb']
        
        try:
            # Download PDB file
            pdb_url = f"{self.base_url}/download/{pdb_id}.pdb"
            async with aiohttp.ClientSession() as session:
                async with session.get(pdb_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download PDB: {response.status}")
                    
                    pdb_content = await response.text()
            
            # Parse structure
            structure_data = self._parse_pdb(pdb_content, pdb_id)
            structure_data['gene'] = gene
            structure_data['pdb_id'] = pdb_id
            
            # Cache for 1 hour
            await cache_manager.set(cache_key, structure_data, ttl=3600)
            
            return structure_data
            
        except Exception as e:
            logger.error(f"Error fetching PDB for {gene}: {str(e)}")
            return None
    
    def _parse_pdb(self, pdb_content: str, pdb_id: str) -> Dict[str, Any]:
        """Parse PDB content and extract structure data"""
        
        # Save to temporary file for parsing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name
        
        try:
            # Parse with BioPython
            structure = self.pdb_parser.get_structure(pdb_id, temp_path)
            
            atoms = []
            colors = []
            chains = {}
            
            # Atom type to color mapping
            atom_colors = {
                'C': [0.5, 0.5, 0.5],  # Gray
                'N': [0.0, 0.0, 1.0],  # Blue
                'O': [1.0, 0.0, 0.0],  # Red
                'S': [1.0, 1.0, 0.0],  # Yellow
                'P': [1.0, 0.5, 0.0],  # Orange
            }
            
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    chains[chain_id] = {'residues': []}
                    
                    for residue in chain:
                        res_id = residue.get_id()[1]
                        res_name = residue.get_resname()
                        
                        chains[chain_id]['residues'].append({
                            'id': res_id,
                            'name': res_name
                        })
                        
                        for atom in residue:
                            coord = atom.get_coord().tolist()
                            atoms.append(coord)
                            
                            element = atom.element.strip() if atom.element else 'C'
                            color = atom_colors.get(element, [0.5, 0.5, 0.5])
                            colors.append(color)
            
            # Calculate secondary structure with DSSP if available
            secondary_structure = self._calculate_secondary_structure(structure)
            
            return {
                'atoms': atoms,
                'colors': colors,
                'chains': chains,
                'secondary_structure': secondary_structure,
                'atom_count': len(atoms),
                'chain_count': len(chains)
            }
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def _calculate_secondary_structure(self, structure) -> Optional[Dict[str, Any]]:
        """Calculate secondary structure using DSSP"""
        try:
            model = structure[0]
            dssp = DSSP(model, '')  # Would need actual DSSP binary
            
            secondary = {
                'helix': [],
                'sheet': [],
                'loop': []
            }
            
            for key in dssp.keys():
                ss_type = dssp[key][2]
                residue_id = key[1][1]
                
                if ss_type == 'H':
                    secondary['helix'].append(residue_id)
                elif ss_type == 'E':
                    secondary['sheet'].append(residue_id)
                else:
                    secondary['loop'].append(residue_id)
            
            return secondary
            
        except Exception as e:
            logger.warning(f"Could not calculate secondary structure: {str(e)}")
            return None
    
    async def fetch_alphafold(self, gene: str) -> Optional[Dict[str, Any]]:
        """Fetch AlphaFold structure"""
        
        mapping = self.GENE_MAPPING.get(gene)
        if not mapping or 'uniprot' not in mapping:
            return None
        
        uniprot_id = mapping['uniprot']
        
        try:
            # Download AlphaFold structure
            af_url = f"{self.alphafold_url}/files/AF-{uniprot_id}-F1-model_v4.pdb"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(af_url) as response:
                    if response.status != 200:
                        return None
                    
                    pdb_content = await response.text()
            
            # Download confidence scores
            conf_url = f"{self.alphafold_url}/files/AF-{uniprot_id}-F1-confidence_v4.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(conf_url) as response:
                    if response.status == 200:
                        confidence_data = await response.json()
                    else:
                        confidence_data = None
            
            structure_data = self._parse_pdb(pdb_content, f"AF-{uniprot_id}")
            
            if confidence_data:
                structure_data['plddt_scores'] = confidence_data
            
            return structure_data
            
        except Exception as e:
            logger.error(f"Error fetching AlphaFold for {gene}: {str(e)}")
            return None
    
    async def fetch_density_map(self, pdb_id: str) -> Optional[str]:
        """Fetch electron density map URL"""
        
        # Return URL to electron density map service
        return f"https://maps.rcsb.org/map/{pdb_id}_2fofc.ccp4"
    
    async def fetch_clinical_variants(self, gene: str) -> List[Dict[str, Any]]:
        """Fetch clinical variants from ClinVar"""
        
        if not os.getenv('ENABLE_CLINVAR', 'True') == 'True':
            return []
        
        try:
            # ClinVar API call
            clinvar_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'clinvar',
                'term': f'{gene}[gene] AND pathogenic[Clinical Significance]',
                'retmode': 'json',
                'retmax': 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(clinvar_url, params=params) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
            
            # Parse ClinVar results
            variants = []
            for variant_id in data.get('esearchresult', {}).get('idlist', []):
                # Get variant details (simplified)
                variants.append({
                    'clinvar_id': variant_id,
                    'gene': gene,
                    'classification': 'Pathogenic'
                })
            
            return variants
            
        except Exception as e:
            logger.error(f"Error fetching ClinVar data: {str(e)}")
            return []
    
    async def annotate_variant(self, structure_data: Dict, variant: Variant) -> Dict:
        """Annotate structure with variant information"""
        
        if not variant.protein_change:
            return structure_data
        
        # Extract position from protein change (e.g., p.Arg175His -> 175)
        import re
        match = re.search(r'\d+', variant.protein_change)
        
        if not match:
            return structure_data
        
        position = int(match.group())
        
        # Modify colors for variant position
        # This is simplified - actual implementation would map residue to atoms
        if 'variants' not in structure_data:
            structure_data['variants'] = []
        
        structure_data['variants'].append({
            'position': position,
            'change': variant.protein_change,
            'classification': variant.classification,
            'color': self._get_variant_color(variant.classification)
        })
        
        return structure_data
    
    def _get_variant_color(self, classification: str) -> List[float]:
        """Get color based on clinical significance"""
        colors = {
            'Pathogenic': [1.0, 0.0, 0.0],  # Red
            'Likely Pathogenic': [1.0, 0.5, 0.0],  # Orange
            'VUS': [1.0, 1.0, 0.0],  # Yellow
            'Likely Benign': [0.0, 1.0, 0.0],  # Green
            'Benign': [0.0, 0.5, 0.0],  # Dark green
        }
        return colors.get(classification, [0.5, 0.5, 0.5])
    
    async def check_drug_binding(self, gene: str, drug: str) -> Optional[DrugInteraction]:
        """Check drug-gene interaction"""
        
        # Simplified drug interaction database
        drug_interactions = {
            ('EGFR', 'Gefitinib'): {
                'interaction_type': 'inhibitor',
                'binding_site': list(range(790, 810)),
                'evidence_level': 'strong',
                'clinical_significance': 'FDA approved for EGFR mutations',
                'recommendation': 'Consider for EGFR L858R or exon 19 deletions'
            },
            ('BRCA1', 'Olaparib'): {
                'interaction_type': 'synthetic_lethality',
                'binding_site': list(range(1650, 1700)),
                'evidence_level': 'strong',
                'clinical_significance': 'FDA approved for BRCA mutations',
                'recommendation': 'Consider for germline BRCA1/2 mutations'
            }
        }
        
        interaction = drug_interactions.get((gene, drug))
        
        if interaction:
            return DrugInteraction(
                gene=gene,
                drug=drug,
                **interaction,
                references=['FDA Label', 'NCCN Guidelines']
            )
        
        return None
    
    async def test_connections(self):
        """Test external API connections"""
        
        tests = {
            'RCSB PDB': self.base_url,
            'AlphaFold': self.alphafold_url,
        }
        
        results = {}
        
        for name, url in tests.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        results[name] = response.status == 200
            except:
                results[name] = False
        
        return results

# ==================== Variant Analysis Service ====================

class VariantAnalysisService:
    """Service for variant analysis"""
    
    def __init__(self):
        self.ncbi_api_key = os.getenv('NCBI_API_KEY')
        
    async def predict_structure_impact(self, variant: Variant) -> Dict[str, Any]:
        """Predict structural impact of variant"""
        
        # Simplified impact prediction
        impact = {
            'stability_change': np.random.uniform(-5, 5),  # ΔΔG in kcal/mol
            'solvent_accessibility': np.random.uniform(0, 1),
            'conservation_score': np.random.uniform(0, 1),
            'functional_domain': self._check_functional_domain(variant),
            'protein_protein_interaction': self._check_ppi_impact(variant)
        }
        
        # Calculate overall impact score
        impact['overall_score'] = self._calculate_impact_score(impact)
        
        return impact
    
    def _check_functional_domain(self, variant: Variant) -> Optional[str]:
        """Check if variant is in functional domain"""
        
        # Simplified domain database
        domains = {
            'BRCA1': {
                (1, 110): 'RING domain',
                (1650, 1863): 'BRCT domain'
            },
            'TP53': {
                (100, 300): 'DNA-binding domain',
                (320, 360): 'Tetramerization domain'
            }
        }
        
        gene_domains = domains.get(variant.gene, {})
        
        # Extract position from protein change
        import re
        match = re.search(r'\d+', variant.protein_change or '')
        if not match:
            return None
        
        position = int(match.group())
        
        for (start, end), domain_name in gene_domains.items():
            if start <= position <= end:
                return domain_name
        
        return None
    
    def _check_ppi_impact(self, variant: Variant) -> bool:
        """Check if variant affects protein-protein interaction"""
        
        # Simplified PPI sites
        ppi_sites = {
            'BRCA1': [61, 1756],  # Known interaction sites
            'TP53': [175, 248, 273]
        }
        
        sites = ppi_sites.get(variant.gene, [])
        
        import re
        match = re.search(r'\d+', variant.protein_change or '')
        if not match:
            return False
        
        position = int(match.group())
        
        return position in sites
    
    def _calculate_impact_score(self, impact: Dict) -> float:
        """Calculate overall impact score"""
        
        score = 0.0
        
        # Stability impact
        if abs(impact['stability_change']) > 2:
            score += 0.3
        
        # Conservation
        score += impact['conservation_score'] * 0.3
        
        # Functional domain
        if impact['functional_domain']:
            score += 0.2
        
        # PPI impact
        if impact['protein_protein_interaction']:
            score += 0.2
        
        return min(score, 1.0)
    
    async def fetch_clinvar(self, variant: Variant) -> Optional[Dict[str, Any]]:
        """Fetch ClinVar annotation for variant"""
        
        try:
            # Build ClinVar query
            query = f"{variant.gene}[gene] AND {variant.protein_change}[Variant]"
            
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'clinvar',
                'term': query,
                'retmode': 'json',
                'api_key': self.ncbi_api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
            
            # Get first result if exists
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if id_list:
                return {
                    'clinvar_id': id_list[0],
                    'found': True,
                    'url': f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{id_list[0]}/"
                }
            
            return {'found': False}
            
        except Exception as e:
            logger.error(f"Error fetching ClinVar: {str(e)}")
            return None

# ==================== ACMG Classifier ====================

class ACMGClassifier:
    """ACMG variant classification service"""
    
    CRITERIA_WEIGHTS = {
        # Pathogenic
        'PVS1': 10,  # Very strong
        'PS1': 7, 'PS2': 7, 'PS3': 7, 'PS4': 7,  # Strong
        'PM1': 4, 'PM2': 4, 'PM3': 4, 'PM4': 4, 'PM5': 4, 'PM6': 4,  # Moderate
        'PP1': 2, 'PP2': 2, 'PP3': 2, 'PP4': 2, 'PP5': 2,  # Supporting
        
        # Benign
        'BA1': -10,  # Stand-alone
        'BS1': -7, 'BS2': -7, 'BS3': -7, 'BS4': -7,  # Strong
        'BP1': -2, 'BP2': -2, 'BP3': -2, 'BP4': -2, 'BP5': -2, 'BP6': -2, 'BP7': -2,  # Supporting
    }
    
    def __init__(self):
        self.models = {}
        
    async def load_models(self):
        """Load ML models for classification"""
        
        try:
            # Load pre-trained models if available
            model_path = os.getenv('ACMG_MODEL_PATH')
            if model_path and os.path.exists(model_path):
                # self.models['classifier'] = joblib.load(model_path)
                logger.info("ACMG classifier model loaded (placeholder)")
        except Exception as e:
            logger.error(f"Error loading ACMG models: {str(e)}")
    
    async def classify(self, variant: Variant) -> Dict[str, Any]:
        """Classify variant using ACMG criteria"""
        
        # Evaluate criteria
        criteria = await self._evaluate_criteria(variant)
        
        # Calculate score
        total_score = sum(
            self.CRITERIA_WEIGHTS.get(c, 0) 
            for c in criteria if c in self.CRITERIA_WEIGHTS
        )
        
        # Determine classification
        if total_score >= 10:
            classification = ClinicalSignificance.PATHOGENIC
        elif total_score >= 6:
            classification = ClinicalSignificance.LIKELY_PATHOGENIC
        elif total_score <= -10:
            classification = ClinicalSignificance.BENIGN
        elif total_score <= -6:
            classification = ClinicalSignificance.LIKELY_BENIGN
        else:
            classification = ClinicalSignificance.VUS
        
        # Calculate confidence
        confidence = min(abs(total_score) / 20, 1.0)
        
        return {
            'classification': classification,
            'acmg_criteria': criteria,
            'score': total_score,
            'confidence': confidence
        }
    
    async def _evaluate_criteria(self, variant: Variant) -> List[str]:
        """Evaluate ACMG criteria for variant"""
        
        criteria = []
        
        # PVS1: Null variant (nonsense, frameshift)
        if self._is_null_variant(variant):
            criteria.append('PVS1')
        
        # PM2: Absent from population databases
        if variant.vaf and variant.vaf < 0.001:
            criteria.append('PM2')
        
        # PP3: Computational prediction
        if await self._check_computational_prediction(variant):
            criteria.append('PP3')
        
        # BP4: Computational prediction benign
        if await self._check_computational_benign(variant):
            criteria.append('BP4')
        
        # Add more criteria evaluation...
        
        return criteria
    
    def _is_null_variant(self, variant: Variant) -> bool:
        """Check if variant is null (nonsense, frameshift)"""
        
        if not variant.protein_change:
            return False
        
        # Check for stop codon (*)
        if '*' in variant.protein_change:
            return True
        
        # Check for frameshift (fs)
        if 'fs' in variant.protein_change.lower():
            return True
        
        return False
    
    async def _check_computational_prediction(self, variant: Variant) -> bool:
        """Check computational predictions (simplified)"""
        
        # In reality, would call multiple prediction tools
        # (SIFT, PolyPhen-2, REVEL, etc.)
        
        # Simplified: random for demo
        return np.random.random() > 0.5
    
    async def _check_computational_benign(self, variant: Variant) -> bool:
        """Check if computational predictions suggest benign"""
        
        return np.random.random() > 0.8

# ==================== Report Generator ====================

class ReportGenerator:
    """Clinical report generation service"""
    
    def __init__(self):
        self.template_path = Path("templates/report_template.html")
        
    async def generate_pdf(self, patient: Dict, variants: List[Dict]) -> str:
        """Generate PDF report"""
        
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        
        # Create PDF file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/report_{patient['patient_id']}_{timestamp}.pdf"
        Path("reports").mkdir(exist_ok=True)
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        story.append(Paragraph("Clinical Genome Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Patient Information
        story.append(Paragraph("Patient Information", styles['Heading2']))
        patient_data = [
            ['Patient ID:', patient['patient_id']],
            ['Name:', patient['name']],
            ['Date of Birth:', patient['birth_date']],
            ['Sex:', patient['sex']],
            ['Test Date:', datetime.now().strftime("%Y-%m-%d")]
        ]
        
        patient_table = Table(patient_data)
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 12))
        
        # Variants
        story.append(Paragraph("Detected Variants", styles['Heading2']))
        
        variant_data = [['Gene', 'Variant', 'Classification', 'VAF']]
        for v in variants:
            variant_data.append([
                v.get('gene', ''),
                v.get('protein_change', ''),
                v.get('classification', ''),
                f"{v.get('vaf', 0)*100:.1f}%"
            ])
        
        variant_table = Table(variant_data)
        variant_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(variant_table)
        story.append(Spacer(1, 12))
        
        # Recommendations
        story.append(Paragraph("Clinical Recommendations", styles['Heading2']))
        
        recommendations = self._generate_recommendations(variants)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        
        return filename
    
    async def generate_json(self, patient: Dict, variants: List[Dict]) -> Dict:
        """Generate JSON report"""
        
        report = {
            'report_id': f"R{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'patient': patient,
            'variants': variants,
            'summary': self._generate_summary(variants),
            'recommendations': self._generate_recommendations(variants),
            'metadata': {
                'version': '1.0',
                'pipeline': 'clinical-genome-visualizer',
                'reference_genome': 'GRCh38'
            }
        }
        
        return report
    
    def _generate_summary(self, variants: List[Dict]) -> str:
        """Generate report summary"""
        
        pathogenic = [v for v in variants if 'Pathogenic' in v.get('classification', '')]
        vus = [v for v in variants if v.get('classification') == 'VUS']
        
        summary = f"Analysis identified {len(variants)} variants: "
        summary += f"{len(pathogenic)} pathogenic/likely pathogenic, "
        summary += f"{len(vus)} variants of uncertain significance."
        
        return summary
    
    def _generate_recommendations(self, variants: List[Dict]) -> List[str]:
        """Generate clinical recommendations"""
        
        recommendations = []
        
        # Check for specific pathogenic variants
        for v in variants:
            if 'Pathogenic' not in v.get('classification', ''):
                continue
            
            gene = v.get('gene')
            
            if gene in ['BRCA1', 'BRCA2']:
                recommendations.append(
                    f"Pathogenic {gene} variant detected. Consider enhanced breast/ovarian "
                    f"cancer screening and risk-reducing surgery consultation."
                )
            elif gene == 'TP53':
                recommendations.append(
                    f"Pathogenic TP53 variant detected. Consider Li-Fraumeni syndrome "
                    f"screening protocol and genetic counseling for family members."
                )
            elif gene == 'MLH1' or gene == 'MSH2':
                recommendations.append(
                    f"Pathogenic {gene} variant detected. Consider Lynch syndrome "
                    f"screening with annual colonoscopy and endometrial surveillance."
                )
        
        if not recommendations:
            recommendations.append(
                "No pathogenic variants detected. Continue standard screening protocols."
            )
        
        return recommendations

# ==================== Main Clinical Analysis Service ====================

class ClinicalAnalysisService:
    """Main service orchestrating clinical analysis pipeline"""
    
    def __init__(self):
        self.structure_service = StructureService()
        self.variant_analysis = VariantAnalysisService()
        self.acmg_classifier = ACMGClassifier()
        self.report_generator = ReportGenerator()
        
    async def initialize(self):
        """Initialize all services"""
        try:
            await self.acmg_classifier.load_models()
            logger.info("Clinical analysis service initialized")
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise
    
    async def analyze_patient(self, patient_data: Dict, variants: List[Variant]) -> Dict[str, Any]:
        """Complete analysis pipeline for a patient"""
        
        logger.info(f"Starting analysis for patient: {patient_data.get('patient_id')}")
        
        results = {
            'patient': patient_data,
            'variants': [],
            'structures': {},
            'drug_interactions': [],
            'report_metadata': {}
        }
        
        # Process each variant
        for variant in variants:
            variant_result = await self._analyze_variant(variant)
            results['variants'].append(variant_result)
            
            # Fetch structure if not already loaded
            if variant.gene not in results['structures']:
                structure = await self.structure_service.fetch_pdb(variant.gene)
                if structure:
                    # Annotate with variant
                    structure = await self.structure_service.annotate_variant(structure, variant)
                    results['structures'][variant.gene] = structure
        
        # Check drug interactions
        for variant in variants:
            drugs = await self._get_relevant_drugs(variant.gene)
            for drug in drugs:
                interaction = await self.structure_service.check_drug_binding(variant.gene, drug)
                if interaction:
                    results['drug_interactions'].append(interaction.__dict__)
        
        # Generate reports
        pdf_path = await self.report_generator.generate_pdf(patient_data, results['variants'])
        json_report = await self.report_generator.generate_json(patient_data, results['variants'])
        
        results['report_metadata'] = {
            'pdf_path': pdf_path,
            'json_report': json_report,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Analysis completed for patient: {patient_data.get('patient_id')}")
        
        return results
    
    async def _analyze_variant(self, variant: Variant) -> Dict[str, Any]:
        """Analyze individual variant"""
        
        # ACMG classification
        acmg_result = await self.acmg_classifier.classify(variant)
        
        # Structure impact prediction
        structure_impact = await self.variant_analysis.predict_structure_impact(variant)
        
        # ClinVar lookup
        clinvar_data = await self.variant_analysis.fetch_clinvar(variant)
        
        return {
            'gene': variant.gene,
            'protein_change': variant.protein_change,
            'genomic_change': variant.genomic_change,
            'vaf': variant.vaf,
            'read_depth': variant.read_depth,
            'classification': acmg_result['classification'],
            'acmg_criteria': acmg_result['acmg_criteria'],
            'confidence': acmg_result['confidence'],
            'structure_impact': structure_impact,
            'clinvar': clinvar_data,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _get_relevant_drugs(self, gene: str) -> List[str]:
        """Get drugs relevant to gene"""
        
        gene_drugs = {
            'EGFR': ['Gefitinib', 'Erlotinib', 'Afatinib'],
            'BRCA1': ['Olaparib', 'Rucaparib'],
            'BRCA2': ['Olaparib', 'Rucaparib'],
            'KRAS': ['Sotorasib'],  # For G12C mutations
            'PIK3CA': ['Alpelisib'],
        }
        
        return gene_drugs.get(gene, [])

# ==================== Service Manager ====================

class ServiceManager:
    """Manager for all backend services"""
    
    def __init__(self):
        self.clinical_service = ClinicalAnalysisService()
        self._initialized = False
        
    async def initialize(self):
        """Initialize all services"""
        if self._initialized:
            return
        
        try:
            await self.clinical_service.initialize()
            self._initialized = True
            logger.info("All services initialized successfully")
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        # Test structure service
        try:
            structure_tests = await self.clinical_service.structure_service.test_connections()
            health['services']['structure'] = {
                'status': 'healthy' if all(structure_tests.values()) else 'degraded',
                'external_apis': structure_tests
            }
        except Exception as e:
            health['services']['structure'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Test cache
        try:
            await cache_manager.set('health_check', 'test', ttl=5)
            cache_result = await cache_manager.get('health_check')
            health['services']['cache'] = {
                'status': 'healthy' if cache_result == 'test' else 'unhealthy'
            }
        except Exception as e:
            health['services']['cache'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Overall status
        service_statuses = [s['status'] for s in health['services'].values()]
        if any(status == 'unhealthy' for status in service_statuses):
            health['status'] = 'unhealthy'
        elif any(status == 'degraded' for status in service_statuses):
            health['status'] = 'degraded'
        
        return health
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service usage statistics"""
        
        stats = {
            'cache_stats': await cache_manager.get_stats(),
            'memory_usage': self._get_memory_usage(),
            'uptime': self._get_uptime(),
        }
        
        return stats
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    def _get_uptime(self) -> str:
        """Get service uptime"""
        
        import psutil
        boot_time = psutil.boot_time()
        uptime_seconds = datetime.now().timestamp() - boot_time
        
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# ==================== Global Service Instance ====================

# Global service manager instance
service_manager = ServiceManager()

# Convenience functions for easy access
async def get_clinical_service() -> ClinicalAnalysisService:
    """Get initialized clinical analysis service"""
    if not service_manager._initialized:
        await service_manager.initialize()
    return service_manager.clinical_service

async def get_structure_service() -> StructureService:
    """Get structure service"""
    clinical_service = await get_clinical_service()
    return clinical_service.structure_service

async def get_variant_analysis_service() -> VariantAnalysisService:
    """Get variant analysis service"""
    clinical_service = await get_clinical_service()
    return clinical_service.variant_analysis

async def get_acmg_classifier() -> ACMGClassifier:
    """Get ACMG classifier"""
    clinical_service = await get_clinical_service()
    return clinical_service.acmg_classifier

async def get_report_generator() -> ReportGenerator:
    """Get report generator"""
    clinical_service = await get_clinical_service()
    return clinical_service.report_generator

# ==================== Error Handling ====================

class ServiceError(Exception):
    """Base exception for service errors"""
    pass

class StructureServiceError(ServiceError):
    """Structure service specific errors"""
    pass

class VariantAnalysisError(ServiceError):
    """Variant analysis specific errors"""
    pass

class ACMGClassificationError(ServiceError):
    """ACMG classification specific errors"""
    pass

class ReportGenerationError(ServiceError):
    """Report generation specific errors"""
    pass

# ==================== Configuration ====================

class ServiceConfig:
    """Configuration for all services"""
    
    # External API settings
    RCSB_PDB_API = os.getenv('RCSB_PDB_API', 'https://files.rcsb.org')
    ALPHAFOLD_API = os.getenv('ALPHAFOLD_API', 'https://alphafold.ebi.ac.uk')
    NCBI_API_KEY = os.getenv('NCBI_API_KEY')
    
    # Feature flags
    ENABLE_CLINVAR = os.getenv('ENABLE_CLINVAR', 'True') == 'True'
    ENABLE_ALPHAFOLD = os.getenv('ENABLE_ALPHAFOLD', 'True') == 'True'
    ENABLE_ML_PREDICTIONS = os.getenv('ENABLE_ML_PREDICTIONS', 'True') == 'True'
    
    # Model paths
    ACMG_MODEL_PATH = os.getenv('ACMG_MODEL_PATH')
    STRUCTURE_MODEL_PATH = os.getenv('STRUCTURE_MODEL_PATH')
    
    # Timeouts and limits
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))
    MAX_VARIANTS_PER_ANALYSIS = int(os.getenv('MAX_VARIANTS_PER_ANALYSIS', '100'))
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))

# ==================== Logging Configuration ====================

def setup_logging():
    """Setup logging configuration"""
    
    logging_level = os.getenv('LOGGING_LEVEL', 'INFO').upper()
    
    logging.basicConfig(
        level=getattr(logging, logging_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/backend_services.log')
        ]
    )
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("Backend services logging configured")

# Initialize logging when module is imported
setup_logging()
