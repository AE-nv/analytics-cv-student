# For code reproducibility we create a training and validation dataset
import os
import torch
from shutil import copyfile
from sklearn.model_selection import train_test_split

image_path_all = "data/images/"
splitted_dataset= "data/images_train/"
validation_size= 0.2
os.makedirs(splitted_dataset,exist_ok=True)
os.makedirs(os.path.join(splitted_dataset,"train"),exist_ok=True)
os.makedirs(os.path.join(splitted_dataset,"val"),exist_ok=True)

data_folders= os.listdir(image_path_all)

def run_split():
  for data_folder in data_folders:
    if (data_folder == ".ipynb_checkpoints"):
      continue
    print("Your folder:", data_folder)
    images = os.listdir(os.path.join(image_path_all,data_folder))
    validation_set_size= int(validation_size *len(images))
    train_dataset, test_dataset = train_test_split(images, test_size=validation_size, random_state=7,shuffle=True)

    os.makedirs(os.path.join(splitted_dataset, "train",data_folder),exist_ok=True)
    os.makedirs(os.path.join(splitted_dataset, "val",data_folder),exist_ok=True)

    for image in train_dataset:
        src_path= os.path.join(image_path_all,data_folder,image)
        dst_path= os.path.join(splitted_dataset,"train",data_folder,image)
        copyfile(src_path, dst_path)

    for image in test_dataset:
        src_path= os.path.join(image_path_all,data_folder,image)
        dst_path= os.path.join(splitted_dataset,"val",data_folder,image)
        copyfile(src_path, dst_path)

    with open('data/classes.txt', 'w') as f:
        for data_folder in data_folders:
            if (data_folder == ".ipynb_checkpoints"):
                continue
            f.write(data_folder)