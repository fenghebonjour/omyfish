import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 300) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),           # motion/focus blur
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.15),  # turbid water
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 300) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
