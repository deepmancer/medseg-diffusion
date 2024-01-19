# MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model
> Revolutionizing Medical Image Segmentation with State-of-the-Art (SOTA) Denoising Diffusion Models with PyTorch
Welcome to a groundbreaking journey in medical imaging and bioinformatics with our PyTorch implementation of [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/pdf/2211.00611.pdf). This repository offers a comprehensive, step-by-step tutorial for harnessing the power of generative models in the realm of complex segmentation tasks such as MRI image analysis. Built meticulously from the ground up using the PyTorch framework, our implementation focuses on advanced segmentation techniques to identify and analyze tumors and cancerous cells. Particularly beneficial for researchers and practitioners in the field, this project also contributes significantly to the understanding of TCGA data, providing an innovative approach to medical image segmentation.

> Advancing the Frontiers of Medical Image Segmentation with Cutting-Edge Diffusion Models in PyTorch
Embark on an academically enriching journey through the realms of medical imaging and bioinformatics with our PyTorch-based implementation of [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/pdf/2211.00611.pdf). This repository is a scholarly endeavor, offering a detailed, methodologically sound tutorial for leveraging generative models in complex segmentation tasks, including MRI image analysis, optic cup segmentation in fundus photography, and thyroid nodule detection in ultrasound imagery. Conceived and developed with meticulous attention to detail using the PyTorch framework, this implementation epitomizes advanced segmentation techniques for precise identification and analysis of tumorous growths and cancerous anomalies. It stands as a pivotal contribution to the field, enhancing the comprehension of TCGA data and offering novel insights into the intricacies of medical image segmentation. This work is an invaluable resource for researchers and practitioners in medical imaging, encapsulating both the theoretical underpinnings and practical applications of diffusion probabilistic models in a wide array of medical segmentation challenges.

This repository offers a complete guide from dataset preparation to model training and evaluation, focusing on the brain tumor segmentation task in MRI images, optic cup segmentation in fundus images, and thyroid nodule segmentation in ultrasound images.

## Introduction
DiffusionMedSeg introduces the first diffusion probabilistic model tailored for general medical image segmentation. Leveraging dynamic conditional encoding and a novel Feature Frequency Parser (FF-Parser), the model remarkably enhances segmentation accuracy in diverse medical imaging modalities.

## Installation
```bash
git clone https://github.com/alirezaheidari-cs/DiffusionMedSeg.git
cd DiffusionMedSeg
pip install -r requirements.txt
