import os
vipshome = 'c:\\vips-dev-8.10\\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

restructured_path = "restructured"
crop_path = "restructured_crop"

image_filepaths = glob(f"{restructured_path}/**/*.jpg", recursive=True)
print(glob(f"{crop_path}/**/*.jpg", recursive=True))

def process_images(image_filepath):
    basename = os.path.basename(image_filepath)
    folder_path = image_filepath.split("\\")
    del folder_path[-1]
    folder_path[0] = crop_path
    folder_path = '/'.join(folder_path)


    image = pyvips.Image.new_from_file(image_filepath)
    w, h = image.width, image.height
    w_crop, h_crop = round(.1 * w), round(.1 * h)

    try:
        top, left, width, height = (h_crop, w_crop, w - w_crop * 2, h - h_crop * 2)
        cropped_im = image.crop(left, top, width, height)

        left, top, width, height = cropped_im.find_trim(threshold=100, background=[255, 255, 255])
        cropped_im = cropped_im.crop(left, top, width, height)
        cropped_im.write_to_file(f"{folder_path}/{basename}")
    except:
        pass


Parallel(n_jobs=20)(delayed(process_images)(image_filepath) for image_filepath in tqdm(image_filepaths))
