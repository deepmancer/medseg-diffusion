# MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model

### _Implementation of MedsegDiff paper from scratch utilizing Pytorch._
> Revolutionizing Medical Image Segmentation with State-of-the-Art (SOTA) Denoising Diffusion Models (DPM) in PyTorch
Welcome to a groundbreaking journey in medical imaging and bioinformatics with our PyTorch implementation of [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/pdf/2211.00611.pdf). This repository offers a comprehensive guide and a step-by-step implementation for utilizing generative models in complex medical diagnosis segmentation tasks such as the segmentation of cancerous tumor cells in Brain MRI scans. The implementation has been meticulously designed using PyTorch and provides an in-depth understanding of advanced segmentation techniques for identifying tumor and cancer anomalies.

# Method

<p align="center">
  <img src="https://github.com/alirezaheidari-cs/MedSegDiffusion/assets/59364943/7e24a41f-ab2b-42d0-a757-0143f825a0a1" alt="An illustration of MedSegDiff" height="300"/>
  <br>
  <i>An illustration of MedSegDiff. For clarity, the time step encoding is omitted in the figure.</i>
</p>

**MedSegDiff** introduces the first diffusion probabilistic model tailored for general medical image segmentation. Leveraging dynamic conditional encoding and a novel Feature Frequency Parser (FF-Parser) which learns a Fourier-space feature space, the model remarkably enhances segmentation accuracy in diverse medical imaging modalities. Following the standard implementation of Diffusion Probabilistic Models (DPM), a U-Net architecture is employed for learning. To achieve segmentation, the step estimation function (\epsilon_{\theta}) is conditioned on the raw image prior, described by:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\epsilon_{\theta}(x_t,%20I,%20t)%20=%20D((E_{t}^{I}%20+%20E_{t}^{x},%20t))" alt="Equation 1"/>
</p>

where ![E_t^I](https://latex.codecogs.com/svg.latex?E_{t}^{I}) is the conditional feature embedding (the raw image embedding), and ![E_t^x](https://latex.codecogs.com/svg.latex?E_{t}^{x}) is the segmentation map feature embedding at the current step. These components are combined and forwarded to a U-Net decoder ![D](https://latex.codecogs.com/svg.latex?D) for reconstruction. The step index ![t](https://latex.codecogs.com/svg.latex?t) is integrated with the embeddings and decoder features, leveraging a shared learned lookup table.

The loss of our model is represented by the following equation:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20\mathbb{E}_{mask_0,\epsilon,t}[\lVert%20\epsilon%20-%20\epsilon_\theta(\sqrt{\bar{a}_t}mask_0%20+%20\sqrt{1%20-%20\bar{a}_t}\epsilon,%20I_i,t)%20\rVert^2])" alt="Equation 2"/>
</p>

In each iteration, a random pair of raw image ![I_i](https://latex.codecogs.com/svg.latex?I_i) and segmentation label ![mask_i](https://latex.codecogs.com/svg.latex?mask_i) are sampled for training. The iteration number is sampled from a uniform distribution and ![epsilon](https://latex.codecogs.com/svg.latex?\epsilon) from a Gaussian distribution. The main architecture of the model is a modified ResUNet, which we implement with a ResNet encoder followed by a UNet decoder. ![I](https://latex.codecogs.com/svg.latex?I) and ![x_t](https://latex.codecogs.com/svg.latex?x_t) (noisy mask at the step=![t](https://latex.codecogs.com/svg.latex?t)) are encoded with two individual encoders.


Please find below a description of the components of the U-Net architecture that will be implemented for this task:

### 1) Dynamic Encoding Process

- **FF-Parser Input**: The segmentation map is first input into the Feature Frequency Parser (FF-Parser), which helps to reduce high-frequency noise and refine the feature representation.

<p align="center">
  <img src="https://github.com/alirezaheidari-cs/MedSegDiffusion/assets/59364943/0d1c9927-7fcf-46cd-bfc2-574165bdecad" alt="An illustration of FF-Parser" height="150"/>
  <br>
  <i>An illustration of FF-Parser. FFT denotes Fast Fourier Transform.</i>
</p>

- **Attentive Fusion**: After FF-Parser processing, the denoised feature map is combined synergistically with the prior image embeddings. An attentive-like mechanism is employed to enhance regional attention and feature saliency.
- **Iterative Refinement**: The enriched feature map is then subject to iterative refinement through the FF-Parser and the attentive mechanism, culminating at the bottleneck phase.
- **Bottleneck Convergence**: At this point, the processed feature map is added to the UNet encoder's outputs, resulting in an improved segmentation map that proceeds to the final encoding stage.

### 2) Time Encoding Block
- **Sinusoidal Embedding Calculation**: This step begins with the calculation of sinusoidal timestep embeddings.
- **Initial Processing Layers**: These embeddings are then passed through a linear layer, followed by SiLU activation, and another linear layer.

<p align="center">
  <img src="https://github.com/alirezaheidari-cs/MedSegDiffusion/assets/59364943/46361efb-680b-437d-84ff-b7bf8ac58984" alt="An illustration of time embedding" height="150"/>
</p>

- **Integration into Residual Blocks**: Time features are then integrated into residual blocks. This is achieved either through straightforward spatial addition or via adaptive group normalization.

### 3) Bottleneck Block
- **Embedding Fusion**: At the core of the U-Net, this phase combines the embeddings from the UNet's encoder with those from the Dynamic Encoding, preparing them for the final encoding stage.

### 4) Encoder & Decoder Blocks 
- **Initial Convolution**: The encoder begins with separate initial convolutional layers for the mask and input image, preparing the features for downstream processing. 
- **Residual Blocks**: As a base sub-module, we define each ResNet block as two consecutive convolutional layers with a SiLU activation in between and Group Normalization after each convolutional layer. You can use this module in Down/Up blocks. Also, by removing the residual connection, you can use this block as a convolutional network throughout the network.
- **Attention**: This is also a sub-module consisting of a Layer Normalization, Multi-head Attention, a residual connection, a feed-forward network, and another residual connection.
- **Time Embedding Integration**: Each residual block in the downsampling and upsampling stages integrates time embeddings.


## Results

From top to down are brain-tumor segmentation, opticcup segmentation, and thyroid nodule segmentation, respectively.

<p align="center">
  <img src="https://github.com/alirezaheidari-cs/MedSegDiffusion/assets/59364943/542aa834-2b65-45de-80c1-628865488742" alt="An illustration of Evaluations - figure" height="300"/>
  <br>
  <i>The visual comparison of Top-4 general medical image segmentation methods</i>
</p>


<p align="center">
  <img src="https://github.com/alirezaheidari-cs/MedSegDiffusion/assets/59364943/33618afd-33c7-4e0b-b0bb-843c0d405362" alt="An illustration of Evaluations - table" height="300"/>
  <br>
  <i>The comparison of MedSegDiff with SOTA segmentation methods. The best results are denoted in <b>bold</b>.</i>
</p>



## Review Diffusion Procces

### Forward Diffusion Process
<p align="center">
  <img src="https://github.com/alirezaheidari-cs/MedSegDiffusion/assets/59364943/7cc2d589-2ca1-4379-aa0a-c87af109346f" alt="Forward Diffusion Process" height="150"/>
  <br>
  <i>Gradually transforms a segmentation label into a noisy mask sequence, converging to a Gaussian distribution as time increases.</i>
</p>

### Reverse Diffusion Process
<p align="center">
  <img src="https://github.com/alirezaheidari-cs/MedSegDiffusion/assets/59364943/ed72eee1-cc51-4504-9e7b-894a0cb224c6" alt="Reverse Diffusion Process" height="150"/>
  <br>
  <i>Denoises noisy data by iteratively removing noise added at each step using the Reverse Diffusion Process.</i>
</p>


## Detailed Method

### Dynamic Encoding Process

To achieve segmentation, we condition the step estimation function by using a **Dynamic Conditional Encoding** process. At step \(t\), we add the segmentation map feature embedding and the raw image embedding and send them to a UNet decoder \(D\) for reconstruction. In most conditional DPM, the conditional prior is unique given information. However, medical image segmentation is more challenging due to its ambiguous objects, where lesions or tissues are often difficult to differentiate from the background. Moreover, low-contrast image modalities such as MRI images make it even more challenging. Hence, given only a static image \(I\) as the condition for each step is hard to learn.

### Attention-like Mechanism

We will implement a dynamic conditional encoding for each step to address this issue. On one hand, the raw image contains accurate segmentation target information but is hard to differentiate from the background. On the other hand, the current-step segmentation map contains enhanced target regions but is not accurate. Therefore, integrating the current-step segmentation information \(x_t\) into the conditional raw image encoding for the mutual complement is a reasonable response. To be specific, we will integrate this on the feature level by fusing conditional feature maps and image encoding features through an **attentive-like mechanism**. This process helps the model to localize and calibrate the segmentation dynamically. In particular, two feature maps are first applied layer normalization and multiplied together to get an affinity map. Then we multiply the affinity map with the condition encoding features to enhance the attentive region, which is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A(m_I^k,%20m_x^k)%20=%20(\text{LayerNorm}(m_I^k)%20\otimes%20\text{LayerNorm}(m_x^k))%20\otimes%20m_I^k" alt="Equation 3"/>
</p>

### Fourier-space Features

There is an issue with integrating the embedding of \(x_t\) as it generates additional high-frequency noise. To address this problem, we restrict the high-frequency elements in the features by training an attentive (weight) map with parameters, which is then applied to the features in **Fourier space**. The following is a detailed explanation of this process:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?M%20=%20\mathcal{F}[m]%20\in%20\mathbb{C}^{H%20\times%20W%20\times%20C}" alt="Equation 4"/>
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?A%20\in%20\mathbb{C}^{H%20\times%20W%20\times%20C},%20\%20M'%20=%20A%20\otimes%20M%20\%20(elementwise%20\%20product)" alt="Equation 5"/>
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?m'%20=%20\mathcal{F}^{-1}[M']" alt="Equation 6"/>
</p>

This FF-Parser can be regarded as a learnable version of frequency filters which are widely applied in digital image processing. Different from spatial attention, it globally adjusts the components of specific frequencies. Thus, it can learn to constrain the high-frequency component for adaptive integration.


## Installation
```bash
git clone https://github.com/alirezaheidari-cs/DiffusionMedSeg.git
cd DiffusionMedSeg
pip install -r requirements.txt
