import os
import random
import shutil
from pathlib import Path

train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15
random.seed(1)

data_dir = Path("../data")
categories = [d for d in data_dir.iterdir() if d.is_dir()]

for cat in categories:
	images = list(cat.glob("*"))
	random.shuffle(images)

	n = len(images)
	train_end = int(n * train_ratio)
	val_end = train_end + int(n * val_ratio)

	splits = {
		"train": images[:train_end],
		"val": images[train_end:val_end],
		"test": images[val_end:]
	}

	for split_name, split_files in splits.items():
        	split_class_dir = data_dir / split_name / cat.name
        	split_class_dir.mkdir(parents=True, exist_ok=True)

        	for file in split_files:
            		shutil.move(str(file), split_class_dir / file.name)

print("dataset split")
