# Comparative Analysis of U-Net-based Architectures on Medical Image Segmentation

This repository contains the source code for the project **“Comparative Analysis of U-Net-based Architectures on Medical Image Segmentation.”**  
The purpose of this project is to implement, evaluate, and compare several U-Net variants on medical image segmentation tasks while providing a clean and modular codebase for future research and experimentation.

## Repository Structure

```
.
├── models/             # Implementations of different U-Net variants
├── utils/              # Data loading, preprocessing, metrics, helpers
├── capstone.ipynb      # Experiment notebook (training + evaluation pipeline)
└── README.md           # Project documentation
```

## Project Objectives

- Implement multiple U-Net variants for medical image segmentation.
- Provide a unified and reproducible framework for comparison.
- Offer a modular codebase that supports adding new architectures or datasets.
- Enable transparent benchmarking across different segmentation models.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/haoruo-zhang/Comparative-Analysis-of-U-Net-based-Architectures-on-Medical-Image-Segmentation.git
cd Comparative-Analysis-of-U-Net-based-Architectures-on-Medical-Image-Segmentation
```
### 2. Prepare the Dataset
Modify or use the existing data loaders under `utils/` to match your dataset format.

### 3. Run the Experiment Notebook
Open and run:
```
capstone.ipynb
```

## Extending the Project

You can extend this repository by adding:

- Additional U-Net variants (Attention U-Net, Residual U-Net, Nested U-Net, TransUNet, etc.)
- More datasets and preprocessing pipelines
- Advanced evaluation metrics (Dice, IoU, HD95, etc.)
- Training scripts with CLI arguments
- Logging, checkpoints, and reproducibility configurations

## License

This repository is intended for academic and research use.  
Feel free to fork, modify, and extend it for your own purposes.
