from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
from model.model import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge, Convnext_custom
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

@st.cache
def load_model():
    model = convnext_base(num_classes=21841, pretrained=True, in_22k=True)
    
    return model

@st.cache
def show_output(image, grad):
    model = load_model()
    img = Image.open(image)
    target_layers = [model.stages[-1]]
    targets = [ClassifierOutputTarget(281)]
    
    if grad == 'GradCAM':
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif grad == 'GradCAMPlusPlus':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    elif grad == 'XGradCAM':
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif grad == 'EigenCAM':
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif grad == 'LayerCAM':
        cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif grad == 'EigenGradCAM':
        cam = EigenGradCAM(model=model, target_layers=target_layers, use_cuda=True)

    input_tensor = torch.unsqueeze(transforms.ToTensor()(img),0)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np.array(img)/255., grayscale_cam, use_rgb=True)

    return visualization