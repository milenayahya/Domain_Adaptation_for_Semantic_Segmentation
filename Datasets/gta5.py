import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as v2
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np
from augmentation import augment

#run splitGTA5.py before to split the data into train and val

class gta5(Dataset):
    def __init__(self,mode,aug=None,cropSize=(512,1024)):
        super(gta5, self).__init__()

        self.mode=mode

        #self.root = Path("/content/GTA5/GTA5")  #google colab path
        self.root = Path("./GTA5/GTA5")   #local path
        
        if mode == "train":
            self.images_path = self.root / "images/train"
            self.labels_path = self.root / "labels/train"

        if mode=="val":
            self.images_path = self.root / "images/val"
            self.labels_path = self.root/ "labels/val"

        print("Checking paths:")
        print("Images path:", self.images_path)
        print("Labels path:", self.labels_path)
       

        #mean and std of ImageNet dataset
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.samples=[]
        self.images=[]
        self.labels =[]

        img_files = sorted(self.images_path.glob("*.png"))
        label_files = sorted(self.labels_path.glob("*.png"))

        for img_path, label_path in zip(img_files,label_files):
            with Image.open(img_path).convert('RGB') as img, Image.open(label_path) as label:
                if mode == "train":
                    #i,j,h,w = v2.RandomCrop.get_params(img, cropSize)
                    #img = TF.crop(img,i,j,h,w)
                     #label= TF.crop(label,i,j,h,w)

                    img = TF.resize(img, cropSize)
                    label = TF.resize(label,cropSize)

                    ## data augmentation if training
                    if aug == True:
                        img, label = augment(img,label)
                   
                img_tensor= self.transform(img)
                self.images.append(img_tensor)
            
                label= np.array(label)
                label_copy = 255 * np.ones(label.shape, dtype=np.float32)
                for k, v in self.id_to_trainid.items():
                    label_copy[label == k] = v

                label_tensor= torch.tensor(label_copy,dtype=torch.float32)
                self.labels.append(label_tensor)

            if(len(self.images))==100:
              break
           

        print("DONE processing 100 images and labels")

        self.samples.extend(zip(self.images,self.labels))
        print("GTA5 dataset initialized")

        #selected_labels = random.sample(self.labels, 100)  
        #unique_labels = torch.unique(torch.cat(selected_labels))
        #print("Unique labels in the dataset:", unique_labels)

    def __getitem__(self, idx):
        image= self.images[idx]
        label= self.labels[idx]

        return image,label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":

    train_dataset = gta5("train", aug=True)
    img,lbl = train_dataset[4]
    print(img,lbl)