# MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model
> Revolutionizing Medical Image Segmentation with State-of-the-Art (SOTA) Denoising Diffusion Models (DPM) in PyTorch
Welcome to a groundbreaking journey in medical imaging and bioinformatics with our PyTorch implementation of [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/pdf/2211.00611.pdf). This repository offers a comprehensive guide and a step-by-step implementation for utilizing generative models in complex medical diagnosis segmentation tasks such as the segmentation of cancerous tumor cells in Brain MRI scans. The implementation has been meticulously designed using PyTorch and provides an in-depth understanding of advanced segmentation techniques for identifying tumor and cancer anomalies.

# Methodology

Following the standard implementation of Diffusion Probabilistic Models (DPM), a U-Net architecture is employed for learning. To achieve segmentation, the step estimation function (\epsilon_{\theta}) is conditioned on the raw image prior, described by:
![Equation1](https://latex.codecogs.com/svg.latex?\epsilon_{\theta}(x_t, I, t) = D((E_{t}^{I} + E_{t}^{x}, t)))

where ![E_t^I](https://latex.codecogs.com/svg.latex?E_{t}^{I}) is the conditional feature embedding (the raw image embedding), and ![E_t^x](https://latex.codecogs.com/svg.latex?E_{t}^{x}) is the segmentation map feature embedding at the current step. These components are combined and forwarded to a U-Net decoder ![D](https://latex.codecogs.com/svg.latex?D) for reconstruction. The step index ![t](https://latex.codecogs.com/svg.latex?t) is integrated with the embeddings and decoder features, leveraging a shared learned lookup table.

The loss of our model is represented by the following equation:

![Equation2](https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20\mathbb{E}_{mask_0,\epsilon,t}[\lVert%20\epsilon%20-%20\epsilon_\theta(\sqrt{\bar{a}_t}mask_0%20+%20\sqrt{1%20-%20\bar{a}_t}\epsilon,%20I_i,t)%20\rVert^2])

In each iteration, a random pair of raw image ![I_i](https://latex.codecogs.com/svg.latex?I_i) and segmentation label ![mask_i](https://latex.codecogs.com/svg.latex?mask_i) are sampled for training. The iteration number is sampled from a uniform distribution and ![epsilon](https://latex.codecogs.com/svg.latex?\epsilon) from a Gaussian distribution. The main architecture of the model is a modified ResUNet, which we implement with a ResNet encoder followed by a UNet decoder. ![I](https://latex.codecogs.com/svg.latex?I) and ![x_t](https://latex.codecogs.com/svg.latex?x_t) (noisy mask at the step=![t](https://latex.codecogs.com/svg.latex?t)) are encoded with two individual encoders.

Please find below a description of the components of the U-Net architecture that will be implemented for this task:

## 1) Dynamic Encoding Process

- **FF-Parser Input**: The segmentation map is first input into the Feature Frequency Parser (FF-Parser), which helps to reduce high-frequency noise and refine the feature representation.
- **Attentive Fusion**: After FF-Parser processing, the denoised feature map is combined synergistically with the prior image embeddings. An attentive-like mechanism is employed to enhance regional attention and feature saliency.
- **Iterative Refinement**: The enriched feature map is then subject to iterative refinement through the FF-Parser and the attentive mechanism, culminating at the bottleneck phase.
- **Bottleneck Convergence**: At this point, the processed feature map is added to the UNet encoder's outputs, resulting in an improved segmentation map that proceeds to the final encoding stage.

## 2) Time Encoding Block
- **Sinusoidal Embedding Calculation**: This step begins with the calculation of sinusoidal timestep embeddings.
- **Initial Processing Layers**: These embeddings are then passed through a linear layer, followed by SiLU activation, and another linear layer.
- **Integration into Residual Blocks**: Time features are then integrated into residual blocks. This is achieved either through straightforward spatial addition or via adaptive group normalization.

## 3) Bottleneck Block
- **Embedding Fusion**: At the core of the U-Net, this phase combines the embeddings from the UNet's encoder with those from the Dynamic Encoding, preparing them for the final encoding stage.

## 4) Encoder & Decoder Blocks 
- **Initial Convolution**: The encoder begins with separate initial convolutional layers for the mask and input image, preparing the features for downstream processing. 
- **Residual Blocks**: As a base sub-module, we define each ResNet block as two consecutive convolutional layers with a SiLU activation in between and Group Normalization after each convolutional layer. You can use this module in Down/Up blocks. Also, by removing the residual connection, you can use this block as a convolutional network throughout the network.
- **Attention**: This is also a sub-module consisting of a Layer Normalization, Multi-head Attention, a residual connection, a feed-forward network, and another residual connection.
- **Time Embedding Integration**: Each residual block in the downsampling and upsampling stages integrate time embeddings.

## 1) Dynamic Encoding Process

[Description of the Dynamic Encoding Process]

### Attention-like Mechanism

[Description of the Attention-like Mechanism]

### Fourier-space Features

[Description of the Fourier-space Features Process]



## Introduction
MedSegDiff introduces the first diffusion probabilistic model tailored for general medical image segmentation. Leveraging dynamic conditional encoding and a novel Feature Frequency Parser (FF-Parser) which learns a Fourier-space feature space, the model remarkably enhances segmentation accuracy in diverse medical imaging modalities.

## Installation
```bash
git clone https://github.com/alirezaheidari-cs/DiffusionMedSeg.git
cd DiffusionMedSeg
pip install -r requirements.txt
