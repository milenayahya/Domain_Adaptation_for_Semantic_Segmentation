from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import shutil


def split80_20(root):
    root = Path(root)
    images_path = root/"images"
    labels_path = root/"labels"

    images= sorted(images_path.glob('*.png'))
    labels = sorted(labels_path.glob('*.png')) 
    samples= list(zip(images,labels))

    train_indexes, val_indexes = train_test_split(list(range(len(samples))), test_size=0.20, random_state=42)

    train_samples = [samples[i] for i in train_indexes]
    val_samples = [samples[i] for i in val_indexes]

    return train_samples, val_samples


def move_dir(samples, dir, mode):
    dir = Path(dir)

    images_dir = dir / "images" / mode
    labels_dir = dir / "labels" / mode
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        shutil.move(sample[0],images_dir)
        shutil.move(sample[1], labels_dir)

if __name__== '__main__':
    train_subset, val_subset = split80_20("./GTA5/GTA5")
    move_dir(train_subset, "./GTA5/GTA5", "train")
    move_dir(val_subset, "./GTA5/GTA5", "val")
