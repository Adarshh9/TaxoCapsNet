# TaxoCapsNet: Full-Stack AI Platform for Autism Prediction

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.13.0-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/flask-2.3.3-green.svg)](https://flask.palletsprojects.com)
[![Next.js](https://img.shields.io/badge/next.js-14.0-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-IEEE%20TCBB-red.svg)](https://ieeexplore.ieee.org)

**A complete AI platform combining taxonomy-aware capsule networks with advanced interpretability and modern web interface for autism spectrum disorder prediction from gut microbiome data.**

> **📄 Research Paper**: "TaxoCapsNet: A Taxonomy-Aware Capsule Network for Autism Prediction from Gut Microbiome Profiles"  
> **👨‍💻 Authors**: Adarsh Kesharwani, Tahami Syed, Shravan Tiwari
> **🏛️ Institution**: Thakur College of Engineering and Technology, Mumbai  
> **📅 Year**: 2025

---

## 🚀 Quick Start

### **Backend Setup**
```bash
# Clone repository
git clone https://github.com/Adarshh9/TaxoCapsNet.git
cd TaxoCapsNet

# Install dependencies
pip install -r requirements.txt

# Train TaxoCapsNet model
python main.py --mode train --config config/config.yaml

# Start Flask API server with SHAP analysis
python flask_server.py
```

### **Frontend Setup**
```bash
# Navigate to frontend directory
cd TaxoFront

# Install Node.js dependencies
npm install

# Start development server
npm run dev

# Visit http://localhost:3000
```

### **Complete Pipeline**
```bash
# Run baseline comparison (recreate paper Table I)
python main.py --mode baseline --data data/GSE_df.csv

# Run ablation studies (recreate paper Table II)
python main.py --mode ablation --seeds 42,123,456
```

## 📊 Key Results

| **Metric** | **TaxoCapsNet** |
|------------|-----------------|
| **Accuracy** | **96.29%** |
| **Sensitivity** | **90.91%** |
| **Specificity** | **98.62%** |
| **AUC** | **0.9702** |
| **F1-Score** | **0.9431** |


### 🔬 Ablation Study Results
- **Capsule vs Dense**: +7.4% improvement
- **Taxonomy vs Random**: +2.8% improvement  
- **Hierarchical vs Flat**: +39.4% improvement

---

## 🏗️ Architecture Overview

TaxoCapsNet introduces **taxonomy-aware capsule networks** that respect the biological hierarchy of gut microbiome data:

### 🧬 Key Innovations

1. **📊 Multi-input Design**: Separate processing streams for each phylum group
2. **🔄 Dynamic Routing**: Capsule routing-by-agreement for hierarchical features
3. **🌿 Biological Awareness**: Respects taxonomic relationships in microbiome data
4. **🔍 Advanced Interpretability**: TaxoSHAP for real-time feature explanations
5. **🌐 Web Interface**: Modern React/Next.js frontend for easy interaction

---

## 📁 Project Structure

```
TaxoCapsNet/
├── 📖 README.md                           # This comprehensive guide
├── 🚀 main.py                            # CLI interface for training/evaluation  
├── 🌐 flask_server.py                    # 🆕 Flask API server with SHAP analysis
├── 📦 requirements.txt                    # Dependencies (updated with Flask & SHAP)
├── ⚙️ setup.py                           # Package installation
├── 🔧 config/
│   └── config.yaml                      # Configuration parameters
├── 🧠 src/
│   ├── 💾 data/
│   │   ├── preprocessing.py             # Data loading & CLR transformation
│   │   └── augmentation.py              # Compositional data augmentation
│   ├── 🏗️ models/
│   │   ├── taxocapsnet.py               # 🌟 Main TaxoCapsNet architecture
│   │   ├── baseline_models.py           # RF, XGBoost, CNN, BiLSTM, etc.
│   │   └── ablation_models.py           # TaxoDense, RandomCaps, FlatCaps
│   ├── 🎯 training/
│   │   ├── trainer.py                   # Training pipeline with CI metrics
│   │   └── evaluation.py                # Multi-seed ablation studies
│   ├── 🧠 interpretability/               # 🆕 NEW: SHAP Analysis Module
│   │   ├── taxoshap.py                  # Advanced SHAP explainer for TaxoCapsNet
│   │   └── shap_utils.py                # SHAP visualization utilities
│   ├── 📊 visualization/
│   │   └── plots.py                     # Comprehensive plotting functions
│   └── 🔧 utils/
│       ├── metrics.py                   # Custom metrics with confidence intervals
│       └── taxonomy.py                  # Taxonomy processing utilities
├── 🌐 TaxoFront/                         # 🆕 Next.js Frontend
│   ├── package.json                     # Node.js dependencies
│   ├── components/                      # React components for analysis
│   ├── pages/                           # Next.js pages (dashboard, analysis)
│   └── styles/                          # CSS and styling
├── 📂 data/                              # Dataset directory
│   └── raw/                             # Raw microbiome data
├── 📈 results/                           # Generated outputs
│   ├── models/                          # Trained model weights
│   ├── figures/                         # Generated plots
│   └── reports/                         # Evaluation reports
└── 🧪 tests/                            # Unit tests (optional)
```

---

## 📦 Installation

### Prerequisites
- **Python 3.8+** and **Node.js 18+**
- **CUDA GPU** (recommended)
- **8GB+ RAM**

### Backend Dependencies
```bash
pip install -r requirements.txt
```

**New dependencies added:**
- `flask==2.3.3` - API server
- `flask-cors==4.0.0` - CORS support for frontend
- `shap==0.42.1` - Model interpretability
- `pyngrok==6.0.0` - Optional public tunneling

### Frontend Dependencies
```bash
cd TaxoFront
npm install
```

---

## 💻 Usage Guide

### 🔥 Command Line Interface

#### **Model Training & Evaluation**
```bash
# Train TaxoCapsNet with default settings
python main.py --mode train --config config/config.yaml

# Run baseline comparison (recreates paper Table I)
python main.py --mode baseline --data data/GSE_df.csv

# Run ablation studies (recreates paper Table II)
python main.py --mode ablation --seeds 42,123,456

# Evaluate with SHAP analysis
python main.py --mode evaluate --model results/models/taxocapsnet.h5 --shap
```

### 🌐 Flask API Server

#### **Start Server**
```bash
# Local server
python flask_server.py

# With ngrok for public access (optional)
USE_NGROK=True NGROK_TOKEN="your_token" python flask_server.py
```

#### **API Endpoints**
- `GET /health` - Health check
- `GET /samples` - List available test samples  
- `GET /phylums` - Get phylum information
- `POST /analyze` - **Main SHAP analysis endpoint**

#### **Example API Usage**
```bash
# Analyze a sample with SHAP
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"sample_id": "sample_0", "method": "tree", "max_display": 15}'
```

### ⚛️ Frontend Interface

#### **Start Frontend**
```bash
cd TaxoFront
npm run dev  # Development server at http://localhost:3000
npm run build && npm start  # Production build
```

**Frontend Features:**
- Interactive dashboard for model predictions
- Real-time SHAP analysis visualization  
- Phylum-level biological interpretations
- Model comparison interface
- Responsive design for all devices

---

## 🆕 New Features

### 🧠 **TaxoSHAP Interpretability**

Advanced SHAP analysis specifically designed for TaxoCapsNet:

```python
# Initialize SHAP explainer
from src.interpretability.taxoshap import create_taxoshap_explainer

explainer = create_taxoshap_explainer(
    model=trained_model,
    X_test_list=X_test_grouped,
    phylum_names=phylum_names,
    taxonomy_map=taxonomy_map
)

# Analyze sample
results = explainer.explain_sample(sample_data, method='tree')

# Generate biological interpretation
report = explainer.generate_reasoning_report(results)
```

**TaxoSHAP provides:**
- **Feature-level importance** for individual OTUs
- **Phylum-level aggregated analysis** 
- **Biological interpretation** with reasoning
- **Multiple visualization types** (bar plots, distributions, scatter)
- **Web-ready base64 images** for frontend integration

### 🌐 **Flask API with Real-time Analysis**

Production-ready Flask server with comprehensive endpoints:

**Key Features:**
- Real-time SHAP analysis via REST API
- CORS support for frontend integration
- Comprehensive error handling and logging
- Optional ngrok tunneling for public access
- Base64 encoded visualizations for web display

**Response Format:**
```json
{
  "status": "success",
  "prediction_results": {
    "predicted_class": 1,
    "prediction_probability": 0.8742,
    "true_label": 1
  },
  "shap_results": {
    "shap_values": [...],
    "feature_names": [...]
  },
  "visualizations": {
    "feature_importance": "iVBORw0KGgoAAAANSU...",
    "phylum_impact": "iVBORw0KGgoAAAANSU..."
  },
  "reasoning_report": {
    "prediction_summary": {...},
    "key_findings": [...],
    "biological_interpretation": [...]
  }
}
```

### 🌐 **Next.js Frontend (TaxoFront)**

Modern web interface for the TaxoCapsNet platform:

**Components:**
- `AnalysisResults.jsx` - Display SHAP analysis results
- `ModelPrediction.jsx` - Prediction interface
- `PhylumAnalysis.jsx` - Biological insights
- `DataUpload.jsx` - File upload for new samples

**Pages:**
- `/` - Main dashboard
- `/analysis` - SHAP analysis interface  
- `/models` - Model comparison
- `/data` - Data management

---

## 🎯 Real-World Applications

### 🏥 **Clinical Decision Support**
- **Early ASD screening** using gut microbiome samples
- **Biomarker identification** for personalized medicine
- **Treatment monitoring** through microbiome changes

### 🔬 **Research Applications**
- **Microbiome-disease association** studies
- **Taxonomic hierarchy analysis** in other datasets
- **Interpretable AI** for biological data

### 🌐 **Platform Deployment**
- **Hospital integration** via REST API
- **Research collaborations** through web interface
- **Large-scale screening** with batch processing

---

## 📊 Models & Evaluation

### 🌟 **TaxoCapsNet** (Main Innovation)
- Multi-input capsule network with dynamic routing
- Taxonomy-aware phylum-level processing
- ~50K parameters, ~70 seconds training time (GPU)

### 🏛️ **Baseline Models**
- **Traditional ML**: Random Forest, XGBoost, Logistic Regression
- **Deep Learning**: CNN, BiLSTM, Transformer, Simple NN
- **Ensemble**: Voting Classifier, Stacking

### 🔬 **Ablation Models**
- **TaxoDense**: Dense networks with taxonomy grouping
- **RandomCapsNet**: Capsules with random OTU grouping  
- **FlatCapsNet**: Capsules without hierarchical input

---

## 🔧 Configuration

### **Complete Config Example**
```yaml
# Data Configuration
data:
  path: "data/GSE_df.csv"
  test_size: 0.2
  preprocessing:
    clr_transformation: true
    standardization: true

# Model Architecture  
model:
  taxocapsnet:
    num_primary_capsules: 8
    routing_iterations: 3
  dense:
    hidden_layers: [64, 32, 16]

# Training Parameters
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

# Flask Server (NEW)
server:
  host: '127.0.0.1'
  port: 5000
  cors_enabled: true
  ngrok:
    enabled: false

# SHAP Analysis (NEW)
interpretability:
  shap:
    method: 'tree'
    background_samples: 50
    max_display: 15
```

---

## 🚀 Production Deployment

### **Docker Deployment**
```bash
# Build and run backend
docker build -t taxocapsnet-api .
docker run -p 5000:5000 taxocapsnet-api

# Build and run frontend
cd TaxoFront
docker build -t taxocapsnet-frontend .
docker run -p 3000:3000 taxocapsnet-frontend
```

### **Cloud Deployment**
- **Backend**: AWS ECS, Google Cloud Run, Azure Container Instances
- **Frontend**: Vercel, Netlify, AWS S3 + CloudFront
- **Full-Stack**: Heroku, Railway, DigitalOcean App Platform

---

## 🧪 Testing & Validation

### **Run Tests**
```bash
# Backend tests
python -m pytest tests/ --cov=src

# Frontend tests  
cd TaxoFront && npm test

# API integration tests
python -m pytest tests/test_api.py -v
```

### **Model Validation**
```bash
# Cross-validation
python main.py --mode crossval --folds 5

# Statistical significance testing
python main.py --mode significance --models TaxoCapsNet,RF,XGB
```

---

## 🛠️ Development

### **Adding New Models**
```python
# 1. Create model in src/models/
# 2. Register in BaselineModelFactory
# 3. Add configuration in config.yaml
# 4. Add tests in tests/
```

### **Extending SHAP Analysis**
```python
# Add new visualization in src/interpretability/shap_utils.py
def create_custom_plot(shap_results, save_format='base64'):
    # Implementation
    return plot_base64
```

### **Frontend Development**
```bash
cd TaxoFront
npm run dev  # Start with hot reload
npm run lint  # Check code quality
npm run build  # Production build
```

---

## 🔍 Troubleshooting

### **Common Issues**
```bash
# GPU memory error
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Dependencies conflict
pip install --no-cache-dir -r requirements.txt

# CORS issues (frontend)
# Check server.cors_origins in config.yaml

# SHAP analysis fails
# Falls back to surrogate model automatically
```

---

## 📚 Citation & References

### **Citing This Work**
```bibtex
@article{kesharwani2025taxocapsnet,
  title={TaxoCapsNet: A Taxonomy-Aware Capsule Network for Autism Prediction from Gut Microbiome Profiles},
  author={Kesharwani, Adarsh and Syed, Tahami and Tiwari, Shravan and Deshmukh, Swaleha},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  year={2025},
  publisher={IEEE}
}
```

### **Useful Links**
- **Dataset**: [GSE113690 on NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113690)
- **Paper**: [IEEE Xplore Digital Library](https://ieeexplore.ieee.org)
- **Issues**: [GitHub Issues](https://github.com/yourusername/TaxoCapsNet/issues)

---

## 👨‍💻 Authors & Contact

| **Author** | **Role** | **Contact** |
|------------|----------|-------------|
| **Adarsh Kesharwani** | Lead Developer, Research | akesherwani900@gmail.com |
| **Tahami Syed** | Research, Implementation | syedtahami123@gmail.com |  
| **Shravan Tiwari** | Research, Validation | shravantiwari2811@gmail.com |
| **Swaleha Deshmukh** | Supervision, Research Guidance | swaleha.deshmukh@tcetmumbai.in |

### **Institution**
**Thakur College of Engineering and Technology**  
Department of Artificial Intelligence and Data Science  
Mumbai, Maharashtra, India  
Website: [tcetmumbai.in](https://tcetmumbai.in)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 **Star this repository if you find it useful!**

**Complete AI Platform • Advanced Interpretability • Modern Web Interface**

**Made with ❤️ by the TaxoCapsNet Team**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/TaxoCapsNet?style=social)](https://github.com/yourusername/TaxoCapsNet)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/TaxoCapsNet?style=social)](https://github.com/yourusername/TaxoCapsNet)

[⬆️ Back to Top](#taxocapsnet-full-stack-ai-platform-for-autism-prediction)

</div>
