# MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model 🚀


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter Notebook">
</p>
<p align="center">

Welcome to **MedSegDiff**—a step-by-step implementation of the [MedSegDiff paper](https://arxiv.org/pdf/2211.00611.pdf) from scratch using PyTorch. MedSegDiff is the first Diffusion Probabilistic Model (DPM) specifically designed for general medical image segmentation tasks, setting a new standard in the identification and segmentation of tumors and cancer anomalies.

---

## 📖 Overview

**MedSegDiff** leverages Diffusion Probabilistic Models (DPM) to advance medical image segmentation. By integrating dynamic conditional encoding and a novel Feature Frequency Parser (FF-Parser) that operates in the Fourier domain, this model significantly improves segmentation accuracy across various medical imaging modalities.

## ⚙️ Methodology

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/model_overview.png" alt="MedSegDiff Overview" height="300"/>
  <br>
  <i>An illustration of MedSegDiff. The time step encoding is omitted for clarity.</i>
</p>

At its core, MedSegDiff utilizes a U-Net architecture for learning and segmentation tasks. The step estimation function is conditioned on the raw image prior, described by:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\epsilon_{\theta}(x_t,%20I,%20t)%20=%20D((E_{t}^{I}%20+%20E_{t}^{x},%20t))" alt="Equation 1"/>
</p>

Here, $\mathbf{E_t^I}$ represents the conditional feature embedding (raw image embedding), and $\mathbf{E_t^x}$ is the segmentation map feature embedding at the current step. These embeddings are combined and processed through a U-Net decoder for reconstruction. The process is governed by the loss function:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20\mathbb{E}_{mask_0,\epsilon,t}[\lVert%20\epsilon%20-%20\epsilon_\theta(\sqrt{\bar{a}_t}mask_0%20+%20\sqrt{1%20-%20\bar{a}_t}\epsilon,%20I_i,t)%20\rVert^2])" alt="Equation 2"/>
</p>

The architecture primarily employs a modified ResUNet, integrating a ResNet encoder with a UNet decoder, offering enhanced segmentation capabilities through its innovative design.

### 🧠 Dynamic Encoding Process

1. **FF-Parser Input**: The segmentation map undergoes initial processing through the Feature Frequency Parser (FF-Parser), which refines feature representation by reducing high-frequency noise.

   <p align="center">
     <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/ff_parser.png" alt="FF-Parser Illustration" height="220"/>
     <br>
     <i>Illustration of the FF-Parser. FFT denotes Fast Fourier Transform.</i>
   </p>

2. **Attentive Fusion**: The denoised feature map is then combined with prior image embeddings using an attentive-like mechanism to enhance regional attention and feature saliency.

3. **Iterative Refinement**: This enriched feature map undergoes further refinement, culminating at the bottleneck phase.

4. **Bottleneck Convergence**: Finally, the processed feature map is integrated with the U-Net encoder's outputs, resulting in an improved segmentation map.

### ⏳ Time Encoding Block

- **Sinusoidal Embedding Calculation**: Sinusoidal timestep embeddings are calculated and passed through a linear layer, followed by SiLU activation, and another linear layer.

   <p align="center">
     <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/time_pe.png" alt="Time Embedding Illustration" height="200"/>
   </p>

- **Integration into Residual Blocks**: Time features are integrated into residual blocks, enhancing the overall model architecture.

### 🛠️ Encoder & Decoder Blocks

- **Initial Convolution**: Separate initial convolutional layers process the mask and input image, preparing them for downstream tasks.
  
- **Residual Blocks**: Each ResNet block, defined by two consecutive convolutional layers with SiLU activation and Group Normalization, is employed throughout the network. Removing the residual connection transforms this block into a basic convolutional network.

- **Attention Mechanism**: A sub-module combining Layer Normalization, Multi-head Attention, residual connections, and a feed-forward network, all crucial for precise segmentation.

### 🔄 Review of Diffusion Process

- **Forward Diffusion Process**: Gradually transforms a segmentation label into a noisy mask sequence, converging to a Gaussian distribution as time increases.

   <p align="center">
     <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/diff_forward.png" alt="Forward Diffusion Process" height="150"/>
   </p>

- **Reverse Diffusion Process**: Iteratively denoises the noisy data, removing the noise added at each step using the Reverse Diffusion Process.

   <p align="center">
     <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/diff_reverse.png" alt="Reverse Diffusion Process" height="150"/>
   </p>

## 🎯 Results

Our method demonstrates superior performance across multiple segmentation tasks, including brain tumor segmentation, optic cup segmentation, and thyroid nodule segmentation.

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/compare.png" alt="Evaluation Results" height="300"/>
  <br>
  <i>Visual comparison of top general medical image segmentation methods.</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/eval.png" alt="SOTA Comparison" height="300"/>
  <br>
  <i>Comparison of MedSegDiff with state-of-the-art segmentation methods. The best results are highlighted in <b>bold</b>.</i>
</p>

## 🚀 Installation

To get started with MedSegDiff, follow these simple steps:

```bash
git clone https://github.com/alirezaheidari-cs/DiffusionMedSeg.git
cd DiffusionMedSeg
pip install -r requirements.txt
```

## 📚 Citations

If you find this work helpful, please consider citing the following papers:

```bibtex
@article{Wu2022MedSegDiffMI,
    title   = {MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model},
    author  = {Junde Wu and Huihui Fang and Yu Zhang and Yehui Yang and Yanwu Xu},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2211.00611}
}
```

```bibtex
@inproceedings{Hoogeboom2023simpleDE,
    title   = {simple diffusion: End-to-end diffusion for high resolution images},
    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
    year    = {2023}
}
```

## 📝 License

This project is licensed under the MIT License. For detailed information, please refer to the [LICENSE](LICENSE) file.

---

We hope this repository aids your research and development in medical image segmentation. Feel free to contribute or raise issues. Let's advance medical technology together! 💡
