# 🧬 Rx-Interactome  
## Predicting Protein Interactomes and Drug Effects on Interaction Networks

**Project Type:** Network Biology + Foundation Model Fine-Tuning  

---

## 📌 Overview

Rx-Interactome is a network-aware biological modeling system designed to predict protein–protein interaction networks (interactomes) and analyze how drugs affect these networks.

Instead of studying single proteins in isolation, this project models biology at the systems level. Diseases often arise from disrupted protein interaction networks rather than individual protein failures. This platform predicts interaction sub-networks and reasons over drug-induced network perturbations using fine-tuned foundation models.

---

## 🧠 Core Idea

- Proteins function through interactions  
- Diseases emerge from network-level disruptions  
- Drugs influence multiple proteins, causing cascading effects  
- Network-aware modeling is essential for modern drug discovery  

---

## 🗂 Data Sources

Biological data was curated from trusted public resources:

- Subcellular Localization Data – Protein compartments  
- Reactome – Pathways and biological processes  
- STRING Database

These sources provide complementary context:
- Where proteins are located  
- Which pathways they belong to  
- How they interact  


## ⚙️ Model Architecture

### 🔹 MedGemma-4B-it
- Predicts protein–protein interactions  
- Constructs interaction sub-networks  

---

## 🧪 Fine-Tuning Strategy

- Supervised Fine-Tuning (SFT)  
- LoRA adapters for parameter-efficient training  
- 4-bit quantization (NF4)  
- bfloat16 precision  
- Structured biological inputs (location, pathways, annotations)  

Training focused on biologically meaningful 3–4 protein interaction groups extracted from larger networks.

---

## 💻 User Interface & Workflow

The application provides an end-to-end pipeline:

1. Upload protein sequences  
2. Predict protein interactions  
3. Generate interaction networks  
4. Analyze drug effects on the network  
5. Visualize network changes  

This makes the system accessible for biological research and hypothesis generation.

---

## 🚀 Technical Highlights

- Built on pretrained health-focused foundation models  
- Efficient LoRA-based adaptation  
- Memory-optimized training with 4-bit quantization  
- Modular architecture for scalable deployment  
- Reproducible training and inference pipeline  

---

## 🔬 Practical Applications

- Systems biology research  
- Drug repurposing  
- Mechanism-of-action analysis  
- Network-based disease modeling  
- Precision medicine research  

---

## 📁 Repository Structure
```
rx-interactome/
│
├── app/                         # Application interface for interactome reasoning
│
├── plots/                       # Visualization outputs and analysis plots
│   └── plots.py                 # Script for generating network and evaluation plots
│
├── data_processing.py           # Core data processing utilities
├── data_prep.py                 # Dataset preparation pipeline
│
├── medgemma_fine_tune.py        # MedGemma LoRA fine-tuning script
├── tx_gemma_fine_tune.py        # TxGemma fine-tuning script
│
├── data_generation_prompt.txt   # Prompt template (version 1)
├── data_generation_prompt_2.txt # Prompt template (version 2)
│
├── download_files.txt           # Required dataset file list
├── download_script.sh           # Automated dataset download script
├── extract_file.sh              # Dataset extraction utility
│
├── requirements.txt             # Python dependencies
├── LICENSE                      # Project license
│
└── README.md                    # Project documentation
```
