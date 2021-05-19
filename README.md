# Image Inpainting with Generative Adversarial Network

# Installation
Clone this repo.
```
git clone https://github.com/LexieYang/CIS735_Project_Image_Inpainting.git
```
# Prerequisites

- Python3
- Pytorch >=1.0
- Tensorboard
- Torchvision
- pillow
# Dataset Preparation
You can download the dataset from the link: https://drive.google.com/drive/folders/1G1fniOxAtcNYrJiNDjMdfBL5pYw_zsmL?usp=sharing

# Training Models
```bash
# To train the model, for example.
python train.py
```
Pre-trained weights and test model
```bash
# To test the model
python test.py
```
You can download the pretrained model from here and save it in a new directory, model_weights/. You can check the qualitative results under the Visual/test/ folder.

# Code Structure
- train.py: the entry point for training.
- test.py: the entry point for testing.
- Experiments/configs.py: defines the configurations
- Dataset/CelabA/face_mask/: contains files that record the image IDs for training, testing and evaluation.
- Dataset/datasets.py: defines the dataloader.
- Model/loss.py: defines the loss functions.
- Model/networks.py: defines the architecture of generator and discriminator.
- Model/AttBlocks.py: defines the attention mechanism.
- Model/PSAattenNet.py: defines the loss, model, optimizetion, foward, backward and others.
- Model/spectral_norm.py: defines the spectral normalization.
# Expected Inputs and Outputs
The inputs of the model are images, including images with masks on and mask images. The outputs of the model are images without masks.
