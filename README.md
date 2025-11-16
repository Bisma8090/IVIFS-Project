# Interactive Visible and Infrared Image Fusion and Segmentation (IVIFS)
**Simplified Implementation**
This is a **simplified implementation** of the IVIFS framework from the paper  
*“Interactive Visible and Infrared Image Fusion and Segmentation (IVIFS)”*.
## 📚 Overview
This project demonstrates how visible and infrared (IR) images can be fused to create enhanced, information-rich images and corresponding segmentation maps.  
It reproduces the conceptual design of the IVIFS framework while allowing controllable fusion through:

•	A learnable fusion parameter α 

•	A text-to-parameter module (IFAM)

•	Dual encoders for Visible/IR images

•	AMIM (attention-based modal interaction)

•	A spatial mask M that highlights important regions

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

1. Clone the Repository
```bash
git clone https://github.com/Bisma8090/IVIFS-Simplified.git
```
2. Navigate to the project directory:
 ```bash
cd ivifs_project
   ```
3. Create and activate environment:
 ```bash
python -m venv venv
venv\Scripts\activate   # (Windows)
   ```
4.Install dependencies:
 ```bash
pip install -r requirements.txt
   ```
5. Train CVIFSM:
 ```bash
python -m src.train_cvifsm --data_root data/custom_dataset --save_dir experiments/outputs/run1 --epochs 20 --batch_size 2 --random_alpha
   ```
6. Train IFAM:
 ```bash
python -m src.train_ifam --epochs 50
   ```
7. α-controlled fusion:
 ```bash
python -m src.inference --data_root data/custom_dataset --ckpt experiments/outputs/run1/model_cvifsm.pth --save_dir experiments/outputs/run1
   ```
8. Text-controlled fusion:
 ```bash
python -m src.text_fusion --vis path/to/vis.jpg --ir path/to/ir.jpg --prompt "more infrared" --output fused.png
```
