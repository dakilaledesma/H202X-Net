import os
vipshome = 'c:\\vips-dev-8.10\\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
from PIL import Image
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

restructured_path = "data/h2021/restructured/train/images"
crop_path = "E:/train_bw/images"
image_size = 600

image_filepaths = glob(f"{restructured_path}/**/*.jpg", recursive=True)
# print(glob(f"{restructured_path}/**/*.jpg", recursive=True))


def process_images(image_filepath):
    basename = os.path.basename(image_filepath)
    folder_path = image_filepath.replace(restructured_path, '')
    folder_path = folder_path.split("\\")
    del folder_path[-1]
    folder_path[0] = crop_path
    folder_path = '/'.join(folder_path)
    # print(folder_path)


    image = Image.open(image_filepath).convert('L')
    resized = image.resize((image_size, image_size))
    # w, h = image.width, image.height
    resized.save(f"{folder_path}/{basename}")

Parallel(n_jobs=20)(delayed(process_images)(image_filepath) for image_filepath in tqdm(image_filepaths))
