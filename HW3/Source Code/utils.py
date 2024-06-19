import numpy as np
import random

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,**sample):
        image, mask, trimap = sample['image'], sample['mask'], sample['trimap']
        if np.random.rand() < self.p:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            trimap = np.fliplr(trimap)
        return {'image': image, 'mask': mask, 'trimap':trimap}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,**sample):
        image, mask, trimap = sample['image'], sample['mask'], sample['trimap']
        if np.random.rand() < self.p:
            image = np.flipud(image)
            mask = np.flipud(mask)
            trimap = np.flipud(trimap)
        return {'image': image, 'mask': mask, 'trimap':trimap}
    
class RandomRotation90degree(object):
    def __call__(self,**sample):
        image, mask, trimap = sample['image'], sample['mask'], sample['trimap']
        angle = random.randint(0, 3)
        image = np.rot90(image, -angle)
        mask = np.rot90(mask, -angle)
        trimap = np.rot90(trimap, -angle)
        return {'image': image, 'mask': mask, 'trimap':trimap}
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **sample):
        for transform in self.transforms:
            sample = transform(**sample)
        return sample

def dice_score(pred_mask, gt_mask):
    # Shape:[batch_size, 1, H(W), W(H)]
    return (pred_mask == gt_mask).sum()/(gt_mask.shape[0]*gt_mask.shape[2]*gt_mask.shape[3])
