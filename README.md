# TaxoCapsNets 🧬🧠
**Taxonomy-Aware Capsule Networks for Autism Spectrum Disorder Prediction from Gut Microbiome Data**

---

## 📌 Overview
TaxoCapsNets is a deep learning framework that leverages **taxonomy-aware Capsule Networks** to predict Autism Spectrum Disorder (ASD) from **16S rRNA gut microbiome profiles**.  
Unlike traditional models, TaxoCapsNets incorporates the **microbial taxonomic hierarchy** (phylum/genus levels) and **dynamic routing-by-agreement** to achieve **higher accuracy** and **better interpretability**.

We integrate **SHAP-based explainability** to reveal biologically meaningful insights, identifying key bacterial taxa that influence ASD predictions.

---

## 🚀 Features
- **Custom Capsule Network architecture** for grouped microbiome data.
- **Dynamic routing-by-agreement** for hierarchical feature learning.
- **Taxonomic grouping** (phylum/genus) to preserve biological relationships.
- **SHAP-based explainability** for both feature-level and group-level insights.
- **End-to-end pipeline**: preprocessing → training → evaluation → explainability.
- Integrated **backend & frontend** for interactive visualization.

---

## 📂 Repository Structure
```
dataset/              # Dataset files (OTU tables, taxonomy mapping)
model/                # Model architecture and training scripts
taxoshap-backend/     # Backend for running SHAP analysis & serving model
taxoshap-frontend/    # Frontend for interactive visualizations
training_metrics/     # Saved training logs and performance metrics
demo.webm             # Demo video of the platform
test.py               # Quick test script to run model inference
```

---

## 🛠 Installation
1. **Clone the repository**
```bash
git clone https://github.com/Adarshh9/TaxoCapsNets.git
cd TaxoCapsNets
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up frontend (optional for visualization)**
```bash
cd taxoshap-frontend
npm install
npm run dev 
```

4. **Run backend server**
```bash
cd taxoshap-backend
RUN THE NOTEBOOK
```

---

## 2️⃣ Run inference Test
```bash
python test.py 
```

---

## 📈 Results
* **Accuracy:** 94%
* **ROC AUC:** 0.98
* **Key Insight:** Certain *Firmicutes* and *Bacteroidetes* phyla were found to be strong predictors for ASD.

---

## 🎥 Demo
![Demo](demo.webm)

---

## 📜 Citation
If you use this work in your research, please cite:

---

## 📬 Contact
**Author:** Adarsh Kesharwani  
📧 [akesherwani900@gmail.com](mailto:akesherwani900@gmail.com)  
---

## 🏷 Keywords
`Capsule Networks` `Taxonomy-aware Deep Learning` `SHAP` `Gut Microbiome` `ASD Prediction`
