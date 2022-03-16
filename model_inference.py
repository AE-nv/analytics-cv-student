import torch
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models

class Predicting_model():
    def __init__(self,model_path,classes_path):
        # TODO changes varialbles to input

        self.output_classes_names= []
        with open(classes_path) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                self.output_classes_names.append(line.strip())
                line = fp.readline()
                cnt += 1

        self.device = torch.device('cpu') # Inference is only don on CPU

        print(self.device)

        self.loader = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        # load model from file
        self.pretrained_model= torch.load(model_path)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

    def image_loader(self,image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = self.loader(image).float()
        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        return image  # assumes that you're using GPU

    def predict_image(self,image_name):
        image_tensor= self.image_loader(image_name)
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.pretrained_model(image_tensor)
            _, preds = torch.max(outputs, 1)
            pred_result_name= self.output_classes_names[preds[0]]
            max_probability= torch.max(torch.softmax(outputs,1)) * 100
            print('Predicted label {}s with probability {:.3f}s %'.format(pred_result_name, max_probability))
            return pred_result_name