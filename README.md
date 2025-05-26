# ⚙️ Dual Disentanglement Domain Generalization (DDDG)

🛠️ This repository provides the official implementation of the paper  

**"[MSSP] Dual Disentanglement Domain Generalization Method for Rotating Machinery Fault Diagnosis"**. [Paper Link](https://www.sciencedirect.com/science/article/pii/S088832702500161X) 

It includes the network architecture, loss functions, Training and testing process in intelligent fault diagnosis tasks.

# 🧪 Environment Setup
This project is developed and tested under Python 3.9 with PyTorch ≥ 1.12.

We recommend using `conda` to manage the environment for consistent dependencies.

# 📁 Project Structure
DDDG

├── x_600.mat #  Quick-run example dataset (600 RPM)

├── x_800.mat #  Quick-run example dataset (800 RPM)
├── x_1000.mat #  Quick-run example dataset (1000 RPM)
├── load_data.py # Data loading and FFT

├── construct_loader.py # Dataloader builder

├── module.py # Loss functions and modules

├── main.py # Training and evaluation

└── README.md

# 🚀 Quick Start

We provide a sample dataset and a ready-to-run script so that users can quickly reproduce the results.

1. **Preparing data**.

- Option 1: To help users get started quickly, we provide three pre-segmented sample sets of vibration signals from the JNU dataset, corresponding to three different operating conditions:

- `x_600.mat`
- `x_800.mat`
- `x_1000.mat`

These `.mat` files are already uploaded to this repository root and can be used directly without any preprocessing. They are representative of different operating conditions and fault types.
- Option 2: Download the data manually from: [📥 External download link](https://github.com/CHAOZHAO-1/Machine-Fault-Dataset)

2. **Please modify the dataset loading path in the code (main.py) to match the actual location where you store the .mat files on your machine**.

3. **Clone this repository**.
   
4. **Run the** main.py **file to start training and evaluation**.

# 📄 Citation

You're warmly welcome to cite our work if you find it useful — we truly appreciate it!

```bibtex
@article{zhang2025dual,
  title={Dual disentanglement domain generalization method for rotating Machinery fault diagnosis},
  author={Zhang, Guowei and Kong, Xianguang and Ma, Hongbo and Wang, Qibin and Du, Jingli and Wang, Jinrui},
  journal={Mechanical Systems and Signal Processing},
  volume={228},
  pages={112460},
  year={2025},
  publisher={Elsevier}
}

