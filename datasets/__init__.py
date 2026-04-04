import os
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset import LeavesDataset, resolve_image_path


class LeavesInferenceDataset(Dataset):
    """Unlabeled dataset used for submission prediction."""

    def __init__(self, image_dir: str, csv_path: str, transforms_fn):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f'Image directory does not exist: {image_dir}')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Prediction csv does not exist: {csv_path}')

        df = pd.read_csv(csv_path)
        if 'image' not in df.columns:
            raise ValueError(f'Prediction csv must contain column: image, got {df.columns.tolist()}')

        self.image_dir = image_dir
        self.images = df['image'].tolist()
        self.transforms = transforms_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = resolve_image_path(self.image_dir, image_name)
        from PIL import Image

        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image, image_name


def _build_transforms(image_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def build_dataloaders(cfg: Dict) -> Tuple[Dict[str, DataLoader], Dict]:
    data_cfg = cfg['data']
    loader_cfg = cfg['loader']
    seed = cfg.get('seed', 42)

    image_dir = data_cfg['image_dir']
    train_csv = data_cfg['train_csv']
    predict_csv = data_cfg.get('test_csv', None)
    train_ratio = data_cfg.get('train_ratio', 0.7)
    val_ratio = data_cfg.get('val_ratio', 0.1)
    test_ratio = data_cfg.get('test_ratio', 0.2)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f'train_ratio + val_ratio + test_ratio must be 1.0, got {ratio_sum}')

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Image directory does not exist: {image_dir}')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f'train_csv does not exist: {train_csv}')

    train_df = pd.read_csv(train_csv)
    if not {'image', 'label'}.issubset(train_df.columns):
        raise ValueError('train_csv must contain columns: image,label')

    class_names = sorted(train_df['label'].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    indices = train_df.index.values
    stratify_labels = train_df['label'].values
    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    train_val_labels = train_df.iloc[train_val_idx]['label'].values
    val_ratio_in_train_val = val_ratio / (train_ratio + val_ratio)
    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio_in_train_val,
            random_state=seed,
            shuffle=True,
            stratify=train_val_labels,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio_in_train_val,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    train_tfms, eval_tfms = _build_transforms(data_cfg.get('image_size', 224))

    train_dataset = LeavesDataset(
        root=image_dir,
        mode='train',
        transforms=train_tfms,
        csv_path=train_csv,
        image_dir=image_dir,
        indices=train_idx,
        class_to_idx=class_to_idx,
    )
    val_dataset = LeavesDataset(
        root=image_dir,
        mode='valid',
        transforms=eval_tfms,
        csv_path=train_csv,
        image_dir=image_dir,
        indices=val_idx,
        class_to_idx=class_to_idx,
    )
    test_dataset = LeavesDataset(
        root=image_dir,
        mode='test',
        transforms=eval_tfms,
        csv_path=train_csv,
        image_dir=image_dir,
        indices=test_idx,
        class_to_idx=class_to_idx,
    )

    batch_size = loader_cfg.get('batch_size', 64)
    num_workers = data_cfg.get('num_workers', 4)
    pin_memory = data_cfg.get('pin_memory', True)

    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=loader_cfg.get('shuffle', True),
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    predict_size = 0
    if predict_csv:
        predict_dataset = LeavesInferenceDataset(
            image_dir=image_dir,
            csv_path=predict_csv,
            transforms_fn=eval_tfms,
        )
        loaders['predict'] = DataLoader(
            predict_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        predict_size = len(predict_dataset)

    sample_image, _ = train_dataset[0]
    input_dim = int(sample_image.numel())

    meta = {
        'num_classes': len(class_to_idx),
        'input_dim': input_dim,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'class_names': class_names,
        'split_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset),
        },
        'predict_size': predict_size,
    }
    return loaders, meta
