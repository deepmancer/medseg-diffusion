# MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model
> Revolutionizing Medical Image Segmentation with State-of-the-Art (SOTA) Denoising Diffusion Models (DPM) in PyTorch
Welcome to a groundbreaking journey in medical imaging and bioinformatics with our PyTorch implementation of [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/pdf/2211.00611.pdf). This repository offers a comprehensive guide and a step-by-step implementation for utilizing generative models in complex medical diagnosis segmentation tasks such as the segmentation of cancerous tumor cells in Brain MRI scans. The implementation has been meticulously designed using PyTorch and provides an in-depth understanding of advanced segmentation techniques for identifying tumor and cancer anomalies.

# Methodology

Following the standard implementation of Diffusion Probabilistic Models (DPM), a U-Net architecture is employed for learning. To achieve segmentation, the step estimation function (\epsilon_{\theta}) is conditioned on the raw image prior, described by:

![Equation1](https://latex.codecogs.com/svg.latex?\epsilon_{\theta}(x_t,%20I,%20t)%20=%20D((E_{t}^{I}%20+%20E_{t}^{x},%20t))

where ![E_t^I](https://latex.codecogs.com/svg.latex?E_{t}^{I}) is the conditional feature embedding (the raw image embedding), and ![E_t^x](https://latex.codecogs.com/svg.latex?E_{t}^{x}) is the segmentation map feature embedding at the current step. These components are combined and forwarded to a U-Net decoder ![D](https://latex.codecogs.com/svg.latex?D) for reconstruction. The step index ![t](https://latex.codecogs.com/svg.latex?t) is integrated with the embeddings and decoder features, leveraging a shared learned lookup table.

The loss of our model is represented by the following equation:

![Equation2](https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20\math


## Introduction
MedSegDiff introduces the first diffusion probabilistic model tailored for general medical image segmentation. Leveraging dynamic conditional encoding and a novel Feature Frequency Parser (FF-Parser) which learns a Fourier-space feature space, the model remarkably enhances segmentation accuracy in diverse medical imaging modalities.

## Installation
```bash
git clone https://github.com/alirezaheidari-cs/DiffusionMedSeg.git
cd DiffusionMedSeg
pip install -r requirements.txt
