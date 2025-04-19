import numpy as np
import os
from numpy import load

# change scene_info_0
directory = './third_party/RoMa_minima/data/megadepth/prep_scene_info/'

for filename in os.listdir(directory):
    f_npz = os.path.join(directory, filename)
    data = load(f_npz, allow_pickle=True).item()
    for count, image_path in enumerate(data['image_paths']):
        if image_path is not None:
            if 'Undistorted_SfM' in image_path:
                depth_path = data['depth_paths'][count]
                if depth_path is not None:
                    new_image_path = depth_path.replace('depths', 'imgs').replace('.h5', '.jpg')

                    data['image_paths'][count] = new_image_path

    data['pairs'] = np.asarray(data['pairs'], dtype=object)
    new_file = './third_party/RoMa_minima/data/megadepth/pre_scene_info_no_sfm/' + filename
    np.save(new_file, data)
    print("Saved to ", new_file)
