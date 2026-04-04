import os
from typing import Dict, Iterable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def resolve_image_path(image_dir: str, image_value: str) -> str:
    """Resolve image path robustly for values like '13501.jpg' or 'images/13501.jpg'."""
    image_value = str(image_value).strip()

    if os.path.isabs(image_value):
        return image_value

    candidate = os.path.join(image_dir, image_value)
    if os.path.exists(candidate):
        return candidate

    norm_value = image_value.replace('\\', '/')
    base_name = os.path.basename(os.path.normpath(image_dir))
    prefix = f'{base_name}/'
    if norm_value.startswith(prefix):
        candidate2 = os.path.join(image_dir, norm_value[len(prefix):])
        if os.path.exists(candidate2):
            return candidate2
        return candidate2

    return candidate


class LeavesDataset(Dataset):
    """Custom Leaves Dataset class with backward-compatible interface."""

    def __init__(
        self,
        root,
        mode='train',
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        transforms=None,
        csv_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        indices: Optional[Iterable[int]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            root (str): Backward-compatible root path. In old usage, images and
                train.csv are both under root.
            mode (str): 'train', 'valid', or 'test'.
            train_ratio (float): Training set ratio.
            val_ratio (float): Validation set ratio.
            test_ratio (float): Test set ratio.
            transforms: torchvision transforms.
            csv_path (str, optional): Explicit CSV path with image/label columns.
            image_dir (str, optional): Explicit image directory.
            indices (Iterable[int], optional): Subset indices from loaded CSV.
            class_to_idx (dict, optional): Label mapping from outer pipeline.
        """
        self.root = root
        self.mode = mode
        self.transforms = transforms

        # New optional explicit paths (for train.csv/test.csv split workflow)
        self.csv_path = csv_path or os.path.join(root, 'train.csv')
        self.image_dir = image_dir or root

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f'CSV file not found: {self.csv_path}')
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f'Image directory not found: {self.image_dir}')

        self.data_df = pd.read_csv(self.csv_path)
        required_cols = {'image', 'label'}
        if not required_cols.issubset(set(self.data_df.columns)):
            raise ValueError(f'CSV must contain columns {required_cols}, got {self.data_df.columns.tolist()}')

        if indices is not None:
            subset_df = self.data_df.iloc[list(indices)].reset_index(drop=True)
            self.images = subset_df['image'].values
            self.labels = subset_df['label'].values
        else:
            # Backward-compatible split logic when no indices are provided.
            self.total_len = len(self.data_df)
            self.train_len = int(self.total_len * train_ratio)
            self.val_len = int(self.total_len * val_ratio)
            self.test_len = self.total_len - self.train_len - self.val_len

            if mode == 'train':
                subset_df = self.data_df.iloc[:self.train_len]
            elif mode == 'valid':
                subset_df = self.data_df.iloc[self.train_len:self.train_len + self.val_len]
            elif mode == 'test':
                subset_df = self.data_df.iloc[self.train_len + self.val_len:]
            else:
                raise ValueError("mode must be one of {'train', 'valid', 'test'}")

            self.images = subset_df['image'].values
            self.labels = subset_df['label'].values

        # Create/receive label mapping.
        unique_labels = sorted(set(self.data_df['label']))
        self.class_to_idx = class_to_idx or {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

        print(f'Finished loading {mode} set: {len(self.images)} samples')

    def __getitem__(self, index):
        """Get a single sample"""
        img_path = resolve_image_path(self.image_dir, self.images[index])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'Error loading image {img_path}: {e}') from e

        if self.transforms is not None:
            image = self.transforms(image)

        label_name = self.labels[index]
        if label_name not in self.class_to_idx:
            raise KeyError(f'Label {label_name} not found in class_to_idx mapping.')

        label = self.class_to_idx[label_name]
        return image, label

    def __len__(self):
        """Return dataset size"""
        return len(self.images)


if __name__ == '__main__':
    # Data paths and parameters
    root = 'dataset'
    batch_size = 128

    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = LeavesDataset(root, mode='train', transforms=transforms_train)
    val_dataset = LeavesDataset(root, mode='valid', transforms=transforms_test)
    test_dataset = LeavesDataset(root, mode='test', transforms=transforms_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train batches: {len(train_loader)}')
    print(f'Valid batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')
    print(f'Number of classes: {len(train_dataset.class_to_idx)}')
