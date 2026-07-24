"""
Polyp segmentation datasets (PraNet protocol).

Split
-----
  Train : Kvasir-SEG (900) + CVC-ClinicDB (550) = 1450 images
  Seen  : Kvasir-SEG (100) + CVC-ClinicDB (62)
  Unseen: CVC-ColonDB (380) + ETIS-LaribPolypDB (196) + CVC-300 (60)

Expected directory layout
-------------------------
  data/
    TrainDataset/
      image/   masks/
    TestDataset/
      Kvasir/             images/ masks/
      CVC-ClinicDB/       images/ masks/
      CVC-ColonDB/        images/ masks/
      ETIS-LaribPolypDB/  images/ masks/
      CVC-300/            images/ masks/
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PolypDataset(Dataset):
    """Generic polyp segmentation dataset."""
    
    def __init__(self, img_dir, mask_dir, img_size=352, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        
        # collect image file names
        self.images = sorted([
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ])
        
        print(f"  [{os.path.basename(os.path.dirname(img_dir))}] "
              f"{len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # load image
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # load mask (the extension may differ from the image)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            # try other extensions
            for ext in ['.jpg', '.jpeg', '.bmp', '.tif']:
                alt = os.path.splitext(img_name)[0] + ext
                alt_path = os.path.join(self.mask_dir, alt)
                if os.path.exists(alt_path):
                    mask_path = alt_path
                    break
        
        mask = Image.open(mask_path).convert('L')
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # geometric augmentation (training only)
        if self.augment:
            # random horizontal flip
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # random vertical flip
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            # random rotation by a multiple of 90 degrees
            k = np.random.randint(0, 4)
            if k > 0:
                img = img.rotate(k * 90)
                mask = mask.rotate(k * 90)
        
        # to tensor
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])(img)
        
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)  # binarize
        mask = mask[np.newaxis, ...]  # (1, H, W)
        
        return img, mask, img_name


def get_train_dataset(data_root, img_size=352):
    """Return the 1450-image training set."""
    return PolypDataset(
        img_dir=os.path.join(data_root, 'TrainDataset', 'image'),
        mask_dir=os.path.join(data_root, 'TrainDataset', 'masks'),
        img_size=img_size,
        augment=True
    )


def get_test_datasets(data_root, img_size=352):
    """Return the five test sets as dict[name -> PolypDataset]."""
    test_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 
                  'ETIS-LaribPolypDB', 'CVC-300']
    
    datasets = {}
    for name in test_names:
        img_dir = os.path.join(data_root, 'TestDataset', name, 'images')
        mask_dir = os.path.join(data_root, 'TestDataset', name, 'masks')
        
        if os.path.exists(img_dir):
            datasets[name] = PolypDataset(
                img_dir=img_dir, mask_dir=mask_dir,
                img_size=img_size, augment=False
            )
        else:
            print(f"  [WARNING] {name} not found at {img_dir}")
    
    return datasets
