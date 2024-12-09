# MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Models 🚀

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=Python&logoColor=ffdd54" alt="Python Badge">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter Notebook Badge">
  <img src="https://img.shields.io/github/license/deepmancer/medseg-diffusion?style=for-the-badge" alt="License Badge">
  <img src="https://img.shields.io/github/stars/deepmancer/medseg-diffusion?style=for-the-badge" alt="GitHub Stars Badge">
</p>

**MedSegDiff** is a comprehensive PyTorch implementation of the [MedSegDiff paper](https://arxiv.org/pdf/2211.00611.pdf), presenting the first Diffusion Probabilistic Model (DPM) designed specifically for general medical image segmentation tasks. This repository aims to provide researchers and practitioners with a clear, step-by-step codebase and documentation to facilitate understanding and application of MedSegDiff across various medical imaging modalities.

---

| **Source Code** | **Website** |
|------------------|-------------|
| [github.com/deepmancer/medseg-diffusion](https://github.com/deepmancer/medseg-diffusion) | [deepmancer.github.io/medseg-diffusion](https://deepmancer.github.io/medseg-diffusion/) |

---

## 🌟 Key Features

- **General Medical Image Segmentation**: Tailored to handle diverse medical imaging tasks, including segmentation of brain tumors, optic cups, and thyroid nodules.
- **Dynamic Conditional Encoding**: Implements step-wise adaptive conditions to enhance regional attention during diffusion.
- **Feature Frequency Parser (FF-Parser)**: Leverages Fourier domain operations to reduce high-frequency noise, improving segmentation quality.
- **From-Scratch Implementation**: Offers a clean and well-documented PyTorch codebase for easy learning and experimentation.
- **Community-Friendly**: Welcomes contributions, issues, and discussions to foster community engagement.

---

## 📖 Table of Contents

- MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Models 🚀
  - 🌟 Key Features
  - 📖 Table of Contents
  - 🔍 Overview
  - 🛠️ Methodology
    - 🔧 Dynamic Conditional Encoding
    - ⏳ Time Encoding Block
    - 🏗️ Encoder & Decoder Blocks
    - 🔄 Diffusion Process (Forward & Reverse)
      - 🟢 Forward Diffusion
      - 🔴 Reverse Diffusion
  - 🎯 Results
  - 🚀 Installation & Usage
  - 📁 Repository Structure
  - 📝 License
  - 🙏 Acknowledgments
  - 🌟 Support the Project
  - 📚 Citations

---

## 🔍 Overview

**MedSegDiff** addresses a fundamental challenge in medical imaging: achieving accurate and robust segmentation across various imaging modalities. Building upon the principles of Diffusion Probabilistic Models (DPMs), MedSegDiff introduces innovative techniques like dynamic conditional encoding and the Feature Frequency Parser (FF-Parser) to enhance the model's ability to focus on critical regions, reduce high-frequency noise, and achieve state-of-the-art segmentation results.

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/model_overview.png" alt="MedSegDiff Overview" height="300"/>
  <br>
  <i>An overview of the MedSegDiff architecture. The time step encoding component is omitted for clarity.</i>
</p>

Formally, at each diffusion step, the model estimates:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\epsilon_{\theta}(x_t,%20I,%20t)%20=%20D((E_{t}^{I}%20+%20E_{t}^{x},%20t))" alt="Equation 1"/>
</p>

Here:
- $\mathbf{E_t^I}$: Conditional feature embedding from the input image.
- $\mathbf{E_t^x}$: Feature embedding of the evolving segmentation mask.
- $D$: A U-Net decoder guiding reconstruction.

The training objective:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20\mathbb{E}_{mask_0,\epsilon,t}[\lVert%20\epsilon%20-%20\epsilon_\theta(\sqrt{\bar{a}_t}mask_0%20+%20\sqrt{1%20-%20\bar{a}_t}\epsilon,%20I_i,t)%20\rVert^2])" alt="Equation 2"/>
</p>

This loss encourages the model to accurately predict the noise added at each step, ultimately guiding the segmentation toward a clean, high-quality mask.

---

## 🛠️ Methodology

MedSegDiff employs a U-Net-based architecture enriched with diffusion steps, dynamic conditional encoding, and Fourier-based noise reduction. The key idea is to iteratively refine a noisy segmentation map into a clean, accurate mask using reverse diffusion steps guided by learned conditioning from the original image.

### 🔧 Dynamic Conditional Encoding

1. **Feature Frequency Parser (FF-Parser)**: The segmentation map first passes through the FF-Parser, which utilizes Fourier transforms to filter out high-frequency noise components, thereby refining the feature representation.

   <p align="center">
     <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/ff_parser.png" alt="FF-Parser Illustration" height="220"/>
     <br>
     <i>The FF-Parser integrates FFT-based denoising before feature fusion.</i>
   </p>

2. **Attentive Fusion**: The denoised feature map is then fused with the image embeddings through an attentive mechanism, enhancing regional attention and improving segmentation precision.

3. **Iterative Refinement**: This combined feature undergoes further refinement, culminating in a bottleneck phase that integrates with encoder features.

4. **Bottleneck Integration**: The refined features merge with the encoder outputs, resulting in the final segmentation mask.

### ⏳ Time Encoding Block

- **Sinusoidal Embeddings**: Timestep embeddings are computed using sinusoidal functions, capturing temporal information of the diffusion process.
- **Integration into Residual Blocks**: These time features are injected into the model's residual blocks, providing temporal context at each diffusion step.

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/time_pe.png" alt="Time Embedding Illustration" height="200"/>
</p>

### 🏗️ Encoder & Decoder Blocks

- **Initial Convolutions**: Separate initial convolutional layers process the input image and the segmentation mask.
- **Residual Blocks**: The backbone consists of ResNet-like blocks with convolutional layers, GroupNorm, and activation functions.
- **Attention Mechanisms**: Multi-head attention modules are incorporated to enhance spatial focus on critical regions.

### 🔄 Diffusion Forward & Reverse Processes (Review)

#### 🟢 Forward Diffusion

In the forward diffusion process, Gaussian noise is progressively added to the segmentation mask over a series of timesteps, degrading it into pure noise.

1. **Noise Addition**: Starting from the original segmentation mask $\text{mask}_0$, Gaussian noise is added iteratively at each timestep $t$, controlled by a variance schedule $\beta_t$.

2. **Progressive Degradation**: This process produces a sequence of increasingly noisy masks $\text{mask}_0, \text{mask}_1, \dots, \text{mask}_T$.

3. **Convergence to Noise**: As $T \to \infty$, the mask becomes indistinguishable from pure Gaussian noise.

<p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/diff_forward.png" alt="Forward Diffusion Process" height="150"/>
</p>

#### 🔴 Reverse Diffusion

The reverse diffusion process aims to reconstruct the original segmentation mask from the noisy data by iteratively denoising.

1. **Noise Prediction**: A U-Net is trained to predict the noise added at each timestep, learning a mapping $\epsilon_\theta(\text{mask}_t, t)$.

2. **Stepwise Denoising**: Starting from $\text{mask}_T$, the model refines the mask by subtracting the predicted noise at each timestep, moving backward from $t = T$ to $t = 0$.

3. **Final Reconstruction**: After $T$ steps, the output $\text{mask}_0$ approximates the original segmentation mask.

<p align="center">
   <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/diff_reverse.png" alt="Reverse Diffusion Process" height="150"/>
</p>

---

## 🎯 Results

MedSegDiff demonstrates superior performance across various medical image segmentation tasks, outperforming state-of-the-art methods by a significant margin.

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/compare.png" alt="Evaluation Results" height="300"/>
  <br>
  <i>Visual comparisons with other segmentation methods.</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/medseg-diffusion/main/images/eval.png" alt="Quantitative Results" height="300"/>
  <br>
  <i>Quantitative results comparing MedSegDiff with state-of-the-art methods. Best results are highlighted in <b>bold</b>.</i>
</p>

---

## 🚀 Installation & Usage

### Requirements

- Python 3.8 or higher
- PyTorch
- Other dependencies as specified in [`requirements.txt`](requirements.txt)

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/deepmancer/medseg-diffusion.git
cd medseg-diffusion
pip install -r requirements.txt
```

### Quick Start

- Explore [`MedSegDiff.ipynb`](MedSegDiff.ipynb) for a comprehensive, step-by-step notebook demonstration.
- Adjust hyperparameters and diffusion steps as needed within the notebook.
- To use your own datasets, modify the data loading sections accordingly.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We extend our gratitude to the authors of the [MedSegDiff paper](https://arxiv.org/pdf/2211.00611.pdf) and other referenced works for their valuable research and insights that inspired this implementation.

---

## 🌟 Support the Project

If you find **MedSegDiff** valuable for your research or projects, please consider starring ⭐ this repository on GitHub. Your support helps others discover this work!

---

## 📚 Citations

If you utilize this repository, please consider citing the following works:

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
