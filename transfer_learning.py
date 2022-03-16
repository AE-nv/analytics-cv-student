from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
import copy
import tensorflow as tf
import tempfile
from torch.utils.tensorboard import SummaryWriter
from google.cloud import storage
from datetime import date

plt.ion()   # interactive mode

class Transfer_learning_classification():
    def __init__(self, epochs=25,lr=0.001,momentum=0.9,gamma=0.1,step_size=7,mobilenet=True):

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.params= {}
        self.params["epochs"]= epochs
        self.params["lr"]= lr
        self.params["momentum"]= momentum
        self.params["gamma"]= gamma
        self.params["step_size"]= step_size
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except:
            self.device= "cpu"
        self.params["device"]=self.device
        self.params["mobilenet"]=mobilenet
        self.writer= None


        self.BEST_MODEL_PATH = 'models/model.pth' # This is explicitly saving the model without mlflow support
        classes_path = 'data/classes.txt'

        data_dir = 'data/images_train'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}

        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        self.number_output_classes= len(self.class_names)
        with open(classes_path, 'w') as f:
            for item in self.class_names:
                f.write("%s\n" % item)

    def log_scalar(self,name, value, step):
        """Log a scalar value to TensorBoard"""
        self.writer.add_scalar(name, value, step)

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=1):
        since = time.time()

        sess = tf.compat.v1.InteractiveSession()

        output_dir = "log_dir"
        print("Writing TensorFlow events locally to %s\n" % output_dir)
        self.writer = SummaryWriter(output_dir)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = float(running_loss / self.dataset_sizes[phase])
                epoch_acc = float(running_corrects.double() / self.dataset_sizes[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == "train":
                    self.log_scalar('train_loss', epoch_loss, epoch)
                    self.log_scalar('train_accuracy', epoch_acc, epoch)

                else:
                    self.log_scalar('test_loss', epoch_loss, epoch)
                    self.log_scalar('test_accuracy', epoch_acc, epoch)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()


            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))


            model.load_state_dict(best_model_wts)
            torch.save(model, self.BEST_MODEL_PATH)

            # Save the best model as mlflow object
            # mlflow.pytorch.log_model(model,"model",registered_model_name=('imageTransferLearning_' + str(date.today().strftime("%d/%m/%Y"))))
            # load best model weights
            return model

            # TODO add additional metrics per class?

    def visualize_model(self,model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.class_names[preds[j]]))
                    plt.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def train_evaluate_finetuning(self): # Finetuning the convnet: reset final fully connected layer
        self.params["mechanism"]="finetuning"

        if self.params["mobilenet"]:
            #https://pytorch.org/docs/stable/torchvision/models.html
            model_ft = models.mobilenet_v2(pretrained=True)
            model_ft.classifier[1] = torch.nn.Linear(in_features=model_ft.classifier[1].in_features, out_features=self.number_output_classes)

        else:
            model_ft = models.resnet18(pretrained=True)  # TODO change here to another network if necessary
            #model_ft = models.mobilenet_v2(pretrained=True) #https://pytorch.org/docs/stable/torchvision/models.html

            num_ftrs = model_ft.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            model_ft.fc = nn.Linear(num_ftrs, self.number_output_classes)

        model_ft = model_ft.to(self.device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=self.params["lr"], momentum=self.params["momentum"])

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=self.params["step_size"], gamma=self.params["gamma"])

        model_ft = self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=self.params["epochs"])

    def train_evaluate_feature_extractor(self):  # Using the convnet as a feature extractor
        self.params["mechanism"]="feature_extractor"

        criterion = nn.CrossEntropyLoss()

        if self.params["mobilenet"]:
            # https://pytorch.org/docs/stable/torchvision/models.html
            model_ft = models.mobilenet_v2(pretrained=True)
            for param in model_ft.parameters():
                param.requires_grad = False

            model_ft.classifier[1] = torch.nn.Linear(in_features=model_ft.classifier[1].in_features,out_features=self.number_output_classes)
            model_ft = model_ft.to(self.device)

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(model_ft.classifier[1].parameters(), lr=self.params["lr"], momentum=self.params["momentum"])

        else:
            model_ft = models.resnet18(pretrained=True)  # Change here to another network if necessary
            for param in model_ft.parameters():
                param.requires_grad = False

            num_ftrs = model_ft.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            model_ft.fc = nn.Linear(num_ftrs, self.number_output_classes)
            model_ft = model_ft.to(self.device)

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=self.params["lr"], momentum=self.params["momentum"])

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=self.params["step_size"],
                                               gamma=self.params["gamma"])

        model_ft = self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                    num_epochs=self.params["epochs"])
