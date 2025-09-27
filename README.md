# Source Separation with Dual Encoders + Transformer Fusion

## Overview
This project explores **music source separation**, with a focus on **vocal isolation**.  
The model is designed as a **dual-encoder architecture** that processes both:
- Raw **waveform input**, and  
- **Spectrogram representations**,  

and then fuses them using a **Transformer-based fusion network** followed by an **MLP decoder**.

The full training pipeline is implemented, including:
- Data preprocessing  
- Model architecture (encoders, fusion, decoder)  
- Loss functions  
- Optimizer and scheduler  
- Checkpointing & validation loop  

**Note**: The project is currently unfinished. The architecture and training pipeline are complete, but due to hardware and memory limitations, full training and evaluation have not yet been carried out.  

---

## Goals
- Separate vocals from mixed audio tracks.  
- Compare dual-encoder fusion against single-representation baselines.  
- Explore Transformer fusion for cross-representation learning.  

---

## Current Status
✅ Data preprocessing implemented  
✅ Dual encoders (waveform + spectrogram)  
✅ Transformer-based fusion network  
✅ MLP decoder  
✅ Training + validation pipeline (with checkpoints)  
⚠️ Training not completed due to hardware limitations  

---

## Tech Stack
- **Python 3.10+**  
- **PyTorch** (deep learning framework)  
- **Librosa** (audio processing)  
- **NumPy / SciPy** (signal processing)  
- **Matplotlib** (visualization)  

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/KenanKhauto/source_seperation.git
   cd source_seperation
---
   To start training run python **"script.py"**.
   If you want to edit and change training logic you can do that in **"script.py"**.

## Licence
This project is open-sourced under the MIT license.

