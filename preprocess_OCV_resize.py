import os
vipshome = 'c:\\vips-dev-8.10\\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import cv2

restructured_path = "data/h2021/restructured/train/images"
crop_path = "data/h2021/restructured/train_s/images"
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


    image = cv2.imread(image_filepath)
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{folder_path}/{basename}", resized)


Parallel(n_jobs=20)(delayed(process_images)(image_filepath) for image_filepath in tqdm(image_filepaths))
