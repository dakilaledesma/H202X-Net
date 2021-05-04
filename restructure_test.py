import os
import shutil
from glob import glob
from tqdm import tqdm

restructure_path = "data/h2020/restructured_test"

test_path = "data/h2020/test"

image_filepaths = glob(f"{test_path}/**/*.jpg", recursive=True)
for filepath in tqdm(image_filepaths, desc="Moving images"):
    image_fn = os.path.basename(filepath)

    shutil.move(filepath, f"{restructure_path}/{image_fn}")