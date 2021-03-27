import os
vipshome = 'c:\\vips-dev-8.10\\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

restructured_path = "data/test/images"
crop_path = "data/test_zoommix/images"
image_size = 448

image_filepaths = glob(f"{restructured_path}/**/*.jpg", recursive=True)
print(glob(f"{restructured_path}/**/*.jpg", recursive=True))


def process_images(image_filepath):
    basename = os.path.basename(image_filepath)
    folder_path = image_filepath.split("\\")
    del folder_path[-1]
    folder_path[0] = crop_path
    folder_path = '/'.join(folder_path)


    image = pyvips.Image.new_from_file(image_filepath)
    resize = image.thumbnail_image(image_size, height=image_size)
    w, h = image.width, image.height

    try:
        top, left, width, height = ((h - image_size) / 2, (w - image_size) / 2, image_size, image_size)
        cropped_im = image.crop(left, top, width, height)

        out_im = 0.5 * resize + 0.5 * cropped_im
        out_im.write_to_file(f"{crop_path}/{basename}")
    except:
        pass


Parallel(n_jobs=20)(delayed(process_images)(image_filepath) for image_filepath in tqdm(image_filepaths[:2]))
