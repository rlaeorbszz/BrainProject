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
[] Data augmentation (brain mri ): 특징 : 흑백, deformation no
: geometry ? / color ? 
: sharpness (선명도)
[]  Batching process
:multi-threading
Batch
1000 -> 128 model update -> 128 model update// iteration epoch
'''

import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2 
from torch.utils.data import Dataset, DataLoader


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
        if len(x.mode)==1: 
            x = x.convert(mode='RGB')

        x = self.transform(x)
        y = torch.LongTensor([gt]).squeeze() # int -> tensor type
        ret = {
            'x' : x,
            'y' : y
        }
        return ret
if __name__ == '__main__':
    transform = transforms.Compose([

        transforms.Resize((224,224)),
        transforms.ToTensor() # 0~255 -> 0~1 /unnormalization
    ]) # c H W -> cv2 H W C (numpy)

    dataset = BrainMRIDataset('./ARCHIVE/Training',True,transform)
    #multi-thread
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    for data in dataloader:
        print(data['x'].shape)

    '''
    for data in dataset:
        print(data['x'].shape, data['y']) #shape=[channel,height,width]

        np_img = (data['x'] * 255).numpy().astype(np.uint8).transpose(1,2,0) # h w c

        cv2.imwrite('./test.png',np_img)
        break
    '''