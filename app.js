/**
 * Clinical Genome Visualizer - Frontend Application
 * Mol* WebGL Integration
 */

class ClinicalGenomeApp {
    constructor() {
        this.viewer = null;
        this.currentStructure = null;
        this.variants = [];
        this.patient = null;
        this.ws = null;
        this.apiUrl = window.location.origin + '/api';
    }

    // ==================== Initialization ====================
    
    async init() {
        console.log('Initializing Clinical Genome Visualizer...');
        
        // Initialize Mol* viewer
        await this.initMolstar();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Connect WebSocket
        this.connectWebSocket();
        
        // Load default structure
        await this.loadStructure('BRCA1');
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        this.showNotification('Application ready', 'success');
    }

    async initMolstar() {
        const container = document.getElementById('molstar-viewer');
        
        // Remove loading overlay from container
        const loadingElement = container.querySelector('.loading-overlay');
        if (loadingElement) {
            loadingElement.style.display = 'flex';
        }
        
        // Initialize Mol* viewer
        this.viewer = await molstar.Viewer.create(container, {
            layoutIsExpanded: false,
            layoutShowControls: false,
            layoutShowRemoteState: false,
            layoutShowSequence: true,
            layoutShowLog: false,
            layoutShowLeftPanel: false,
            
            viewportShowExpand: true,
            viewportShowSelectionMode: true,
            viewportShowAnimation: true,
            
            volumeStreamingServer: 'https://maps.rcsb.org',
            pdbProvider: 'rcsb',
            emdbProvider: 'rcsb',
        });
        
        console.log('Mol* viewer initialized');
    }

    // ==================== Event Listeners ====================
    
    setupEventListeners() {
        // Patient form
        document.getElementById('patient-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.savePatient();
        });
        
        // Variant form
        document.getElementById('variant-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addVariant();
        });
        
        // Gene selection change
        document.getElementById('variant-gene').addEventListener('change', (e) => {
            this.loadStructure(e.target.value);
        });
    }

    // ==================== WebSocket ====================
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws/${this.getPatientId()}`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateWebSocketStatus(true);
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateWebSocketStatus(false);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateWebSocketStatus(false);
            
            // Reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }
    
    updateWebSocketStatus(connected) {
        const indicator = document.getElementById('ws-indicator');
        const status = document.querySelector('#ws-status span');
        
        if (connected) {
            indicator.style.background = '#44ff44';
            status.textContent = 'Connected';
        } else {
            indicator.style.background = '#ff4444';
            status.textContent = 'Disconnected';
        }
    }
    
    handleWebSocketMessage(data) {
        console.log('WebSocket message:', data);
        
        // Handle different message types
        switch(data.type) {
            case 'variant_analyzed':
                this.updateVariantsList();
                this.showNotification('Variant analysis complete', 'success');
                break;
            case 'structure_updated':
                this.loadStructure(data.gene);
                break;
            case 'report_ready':
                this.showNotification('Report ready for download', 'info');
                break;
        }
    }

    // ==================== Patient Management ====================
    
    async savePatient() {
        const patientData = {
            patient_id: document.getElementById('patient-id').value,
            name: document.getElementById('patient-name').value,
            birth_date: document.getElementById('patient-birth').value,
            sex: document.getElementById('patient-sex').value,
            clinical_diagnosis: document.getElementById('clinical-diagnosis').value
        };
        
        try {
            const response = await axios.post(`${this.apiUrl}/patients`, patientData);
            this.patient = response.data;
            
            this.showNotification('Patient saved successfully', 'success');
            
            // Send via WebSocket
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'patient_created',
                    data: this.patient
                }));
            }
        } catch (error) {
            console.error('Error saving patient:', error);
            this.showNotification('Failed to save patient', 'error');
        }
    }
    
    getPatientId() {
        return document.getElementById('patient-id').value || 'default';
    }

    // ==================== Variant Management ====================
    
    async addVariant() {
        const variantData = {
            gene: document.getElementById('variant-gene').value,
            protein_change: document.getElementById('variant-change').value,
            vaf: parseFloat(document.getElementById('variant-vaf').value) / 100
        };
        
        // Validate
        if (!variantData.protein_change) {
            this.showNotification('Please enter protein change', 'warning');
            return;
        }
        
        try {
            // Send to backend for analysis
            const response = await axios.post(`${this.apiUrl}/variants/analyze`, {
                ...variantData,
                patient_id: this.getPatientId()
            });
            
            const result = response.data;
            
            // Add to local list
            this.variants.push(result);
            
            // Update UI
            this.updateVariantsList();
            this.updateStatistics();
            
            // Visualize variant
            await this.visualizeVariant(result);
            
            // Clear form
            document.getElementById('variant-change').value = '';
            document.getElementById('variant-vaf').value = '50';
            
            this.showNotification('Variant added and analyzed', 'success');
            
        } catch (error) {
            console.error('Error adding variant:', error);
            this.showNotification('Failed to analyze variant', 'error');
        }
    }
    
    updateVariantsList() {
        const container = document.getElementById('variants-list');
        
        container.innerHTML = this.variants.map(v => `
            <div class="variant-card ${v.classification.toLowerCase().replace(' ', '-')}" 
                 onclick="app.visualizeVariant(${JSON.stringify(v).replace(/"/g, '&quot;')})">
                <div class="variant-gene">${v.variant.gene}</div>
                <div class="variant-change">${v.variant.protein_change}</div>
                <div class="variant-details">
                    <span>VAF: ${(v.variant.vaf * 100).toFixed(1)}%</span>
                    <span>${v.classification}</span>
                </div>
            </div>
        `).join('');
    }
    
    updateStatistics() {
        document.getElementById('total-variants').textContent = this.variants.length;
        
        const pathogenic = this.variants.filter(v => 
            v.classification.includes('Pathogenic')
        ).length;
        const vus = this.variants.filter(v => 
            v.classification === 'VUS'
        ).length;
        const benign = this.variants.filter(v => 
            v.classification.includes('Benign')
        ).length;
        
        document.getElementById('pathogenic-count').textContent = pathogenic;
        document.getElementById('vus-count').textContent = vus;
        document.getElementById('benign-count').textContent = benign;
        
        // Update recommendations
        this.updateRecommendations();
    }
    
    updateRecommendations() {
        const container = document.getElementById('recommendations');
        const pathogenicVariants = this.variants.filter(v => 
            v.classification.includes('Pathogenic')
        );
        
        if (pathogenicVariants.length === 0) {
            container.innerHTML = `
                <div class="alert alert-success">
                    No pathogenic variants detected. Continue standard screening.
                </div>
            `;
            return;
        }
        
        const recommendations = [];
        
        pathogenicVariants.forEach(v => {
            if (v.variant.gene === 'BRCA1' || v.variant.gene === 'BRCA2') {
                recommendations.push(`
                    <div class="alert alert-warning">
                        <strong>${v.variant.gene} ${v.variant.protein_change}</strong>: 
                        High risk for breast/ovarian cancer. 
                        Recommend enhanced screening with annual MRI and consideration of prophylactic surgery.
                    </div>
                `);
            } else if (v.variant.gene === 'TP53') {
                recommendations.push(`
                    <div class="alert alert-error">
                        <strong>TP53 ${v.variant.protein_change}</strong>: 
                        Li-Fraumeni syndrome suspected. 
                        Recommend whole-body MRI screening and genetic counseling.
                    </div>
                `);
            }
        });
        
        container.innerHTML = recommendations.join('');
    }

    // ==================== 3D Visualization ====================
    
    async loadStructure(gene) {
        try {
            document.getElementById('loading').style.display = 'flex';
            
            // Request structure from backend
            const response = await axios.post(`${this.apiUrl}/structure/visualize`, {
                gene: gene,
                show_density: false,
                show_alphafold: false,
                show_clinical_variants: false
            });
            
            const structureData = response.data;
            
            // Clear current structure
            await this.viewer.clear();
            
            // Load new structure
            await this.viewer.loadStructureFromUrl(
                `https://files.rcsb.org/download/${structureData.pdb_id}.pdb`,
                'pdb'
            );
            
            this.currentStructure = structureData;
            
            // Apply default styling
            this.applyDefaultStyling();
            
            document.getElementById('loading').style.display = 'none';
            
        } catch (error) {
            console.error('Error loading structure:', error);
            document.getElementById('loading').style.display = 'none';
            this.showNotification('Failed to load structure', 'error');
        }
    }
    
    async visualizeVariant(variant) {
        if (!this.currentStructure) return;
        
        // Extract position from protein change
        const match = variant.variant.protein_change.match(/\d+/);
        if (!match) return;
        
        const position = parseInt(match[0]);
        
        // Highlight variant position
        // Note: Actual Mol* selection API would be more complex
        this.viewer.visual.highlight({
            struct: this.currentStructure,
            residue: position,
            color: this.getVariantColor(variant.classification)
        });
        
        // Focus camera on variant
        this.viewer.camera.focus({
            target: { residue: position }
        });
    }
    
    getVariantColor(classification) {
        const colors = {
            'Pathogenic': { r: 255, g: 68, b: 68 },
            'Likely Pathogenic': { r: 255, g: 136, b: 0 },
            'VUS': { r: 255, g: 170, b: 0 },
            'Likely Benign': { r: 136, g: 204, b: 0 },
            'Benign': { r: 68, g: 255, b: 68 }
        };
        
        return colors[classification] || { r: 128, g: 128, b: 128 };
    }
    
    applyDefaultStyling() {
        // Apply cartoon representation with spectrum coloring
        // Note: Actual Mol* API calls would be different
        this.viewer.visual.update({
            representationType: 'cartoon',
            colorTheme: 'sequence-id'
        });
    }

    // ==================== Advanced Features ====================
    
    async showDensityMap() {
        if (!this.currentStructure) return;
        
        try {
            this.showNotification('Loading electron density map...', 'info');
            
            // Load density map
            await this.viewer.loadVolumeFromUrl(
                `https://maps.rcsb.org/map/${this.currentStructure.pdb_id}_2fofc.ccp4`,
                'ccp4'
            );
            
            this.showNotification('Density map loaded', 'success');
            
        } catch (error) {
            console.error('Error loading density map:', error);
            this.showNotification('Failed to load density map', 'error');
        }
    }
    
    async showAlphaFold() {
        const gene = document.getElementById('variant-gene').value;
        
        // UniProt mapping
        const uniprotMap = {
            'BRCA1': 'P38398',
            'BRCA2': 'P51587',
            'TP53': 'P04637',
            'EGFR': 'P00533',
            'KRAS': 'P01116',
            'PIK3CA': 'P42336'
        };
        
        const uniprotId = uniprotMap[gene];
        if (!uniprotId) {
            this.showNotification('AlphaFold not available for this gene', 'warning');
            return;
        }
        
        try {
            this.showNotification('Loading AlphaFold structure...', 'info');
            
            await this.viewer.loadStructureFromUrl(
                `https://alphafold.ebi.ac.uk/files/AF-${uniprotId}-F1-model_v4.pdb`,
                'pdb'
            );
            
            // Apply pLDDT confidence coloring
            this.viewer.visual.update({
                colorTheme: 'plddt-confidence'
            });
            
            this.showNotification('AlphaFold structure loaded', 'success');
            
        } catch (error) {
            console.error('Error loading AlphaFold:', error);
            this.showNotification('Failed to load AlphaFold structure', 'error');
        }
    }
    
    async showClinicalVariants() {
        if (!this.currentStructure) return;
        
        try {
            // Fetch clinical variants from backend
            const response = await axios.get(
                `${this.apiUrl}/structure/clinical_variants/${this.currentStructure.gene}`
            );
            
            const clinicalVariants = response.data;
            
            // Highlight all clinical variants
            clinicalVariants.forEach(v => {
                this.viewer.visual.add({
                    type: 'sphere',
                    position: v.position,
                    color: this.getVariantColor(v.classification),
                    radius: 1.5
                });
            });
            
            this.showNotification(`${clinicalVariants.length} clinical variants displayed`, 'info');
            
        } catch (error) {
            console.error('Error loading clinical variants:', error);
            this.showNotification('Failed to load clinical variants', 'error');
        }
    }
    
    async showDrugBinding() {
        const gene = document.getElementById('variant-gene').value;
        
        // FDA approved drugs
        const drugMap = {
            'EGFR': 'Gefitinib',
            'BRCA1': 'Olaparib',
            'BRCA2': 'Olaparib'
        };
        
        const drug = drugMap[gene];
        if (!drug) {
            this.showNotification('No drug data available for this gene', 'warning');
            return;
        }
        
        try {
            const response = await axios.post(`${this.apiUrl}/drug/interaction`, {
                gene: gene,
                drug: drug
            });
            
            const bindingSite = response.data;
            
            // Highlight binding site
            this.viewer.visual.add({
                type: 'surface',
                selection: bindingSite.residues,
                color: { r: 0, g: 255, b: 0 },
                opacity: 0.5
            });
            
            this.showNotification(`${drug} binding site highlighted`, 'success');
            
        } catch (error) {
            console.error('Error showing drug binding:', error);
            this.showNotification('Failed to show drug binding', 'error');
        }
    }
    
    resetView() {
        this.viewer.camera.reset();
        this.applyDefaultStyling();
    }
    
    async takeScreenshot() {
        try {
            const screenshot = await this.viewer.helpers.screenshot();
            
            // Create download link
            const link = document.createElement('a');
            link.download = `structure_${Date.now()}.png`;
            link.href = screenshot;
            link.click();
            
            this.showNotification('Screenshot saved', 'success');
            
        } catch (error) {
            console.error('Error taking screenshot:', error);
            this.showNotification('Failed to take screenshot', 'error');
        }
    }

    // ==================== Report Generation ====================
    
    async generateReport() {
        const patientId = this.getPatientId();
        
        if (!patientId || this.variants.length === 0) {
            this.showNotification('Please add patient and variants first', 'warning');
            return;
        }
        
        try {
            this.showNotification('Generating report...', 'info');
            
            const response = await axios.post(
                `${this.apiUrl}/report/generate`,
                { patient_id: patientId },
                { responseType: 'blob' }
            );
            
            // Create download link
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.download = `report_${patientId}_${Date.now()}.pdf`;
            link.click();
            
            this.showNotification('Report generated and downloaded', 'success');
            
        } catch (error) {
            console.error('Error generating report:', error);
            this.showNotification('Failed to generate report', 'error');
        }
    }
    
    async exportData() {
        const data = {
            patient: this.patient,
            variants: this.variants,
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `patient_data_${Date.now()}.json`;
        link.click();
        
        this.showNotification('Data exported', 'success');
    }

    // ==================== Notifications ====================
    
    showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');
        
        const notification = document.createElement('div');
        notification.className = `alert alert-${type}`;
        notification.style.marginBottom = '10px';
        notification.style.animation = 'slideIn 0.3s ease';
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// CSS Animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);