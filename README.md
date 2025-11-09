# Interactive Visible and Infrared Image Fusion and Segmentation (IVIFS)
**Simplified Implementation**
This is a **simplified Python implementation** of the IVIFS framework from the paper  
*“Interactive Visible and Infrared Image Fusion and Segmentation (IVIFS)”*.
## 📚 Overview
This project demonstrates how visible and infrared (IR) images can be fused to create enhanced, information-rich images and corresponding segmentation maps.  
It reproduces the conceptual design of the IVIFS framework using CPU-friendly Python scripts

---

## 🔹 Folder Structure
ivifs_project/

├─ datasets/ → contains visible and infrared sample images<br>
├─ outputs/ → fused and segmentation results<br>
├─ scripts/ → Python scripts for fusion & interaction

---
## 🧠 Key Components

| Module | Description |
|--------|--------------|
| **CVIFSM** | *Controllable Visible-Infrared Fusion & Segmentation Module* — implemented in `train_cvifsm.py`; fuses visible + IR images with adjustable weight (α). |
| **IRM** | *Interactive Reinforcement Module* — implemented in `irm_interactive.py`; changes α based on user text prompts (e.g., “Enhance infrared details”, “Increase visible contrast”). |
| **Infrared Simulation** | `make_infrared.py` generates synthetic infrared images from visible samples. |
| **Visualization** | `segmentation_visual.py` displays input, fused, and segmentation outputs side-by-side. |
| **Metrics** | `metrics_eval.py` calculates image quality metrics (SD, Entropy, CC) and segmentation accuracy (mIoU). |

---
## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Bisma8090/IVIFS-Simplified.git
cd ivifs_project
