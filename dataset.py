'''

class data load feature







'''

'''
TODO
[] channel 3 convert

[]  Resize into(224,224)
class test_Dataset(Dataset):
    def __init__(self, file_list,transform):
        self.file_list = file_list
        self.transfrom = transform
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img = self.transform(img)
    return img
[] Data augmentation
[]  Batching process

'''

import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

int dir = 3
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, is_train, transform):
        super().__init__()
        self.category_to_int = {'notumor':0,
                                'glioma':1,
                                'meningioma':2,
                                'pituitary':3
        }
        self.sample_path=[]
        self.ground_truth = []
        for dir_name in os.listdir(root_dir):
            if dir_name == '.DS_Store':
                continue
            # ./ARCHIVE/Training/notumor
            category_path = os.path.join(root_dir,dir_name)
            for image_name in os.listdir(category_path):
               

                image_path = os.path.join(category_path, image_name)
                self.sample_path.append(image_path)
                self.ground_truth.append(self.category_to_int[dir_name])

        assert len(self.sample_path) == len(self.ground_truth)

        self.transform = transform
           
       
        #print(root_dir, is_train)

    def __len__(self):
        return len(self.sample_path)
    def __getitem__(self,idx):
        file_path = self.sample_path[idx]
        gt = self.ground_truth[idx]

        x=Image.open(file_path) # pillow type

        x = self.transform(x)
        y = torch.LongTensor([gt]).squeeze() # int -> tensor type
        ret = {
            'x' : x,
            'y' : y
        }
        return ret
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = BrainMRIDataset('./ARCHIVE/Training',True,transform)
    
    for data in dataset:
        print(data['x'].shape, data['y']) #shape=[channel,height,width]
