import os
from PIL import Image

root_directory = './data/megadepth/train/phoenix/S6/zl548/MegaDepth_v1/'

for folder in os.listdir(root_directory):
    four_digit_directory = os.path.join(root_directory,folder)
    for dense_folder in os.listdir(four_digit_directory):
        image_directory =  os.path.join(four_digit_directory,dense_folder,'imgs')
        for image in os.listdir(image_directory):
            if 'JPG' in image:
                new_name = image.replace('JPG', 'jpg')
                old_path = os.path.join(image_directory, image)
                new_path = os.path.join(image_directory, new_name)
                os.rename(old_path, new_path)
            if 'png' in image:
                new_name = image.replace('png', 'jpg')

                old_path = os.path.join(image_directory, image)
                new_path = os.path.join(image_directory, new_name)
                png_img = Image.open(old_path)
                print('old_path:', old_path)
                png_img.save(new_path)