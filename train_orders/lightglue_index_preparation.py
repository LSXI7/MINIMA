import argparse
import numpy as np
import os
from numpy import load


def index_preparation(directory, txt_path):
    for filename in os.listdir(directory):
        f_npz = os.path.join(directory, filename)

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        base_filename = os.path.splitext(filename)[0]
        if base_filename not in lines:
            continue

        data = load(f_npz, allow_pickle=True)
        data_dict = dict(data)
        image_paths = list(data_dict['image_paths'])

        for count, image_path in enumerate(image_paths):
            if image_path is not None:
                print('image_path:', image_path)
                if 'Undistorted_SfM' in image_path:
                    # print(data_dict['depth_paths'][count])
                    # print(image_paths[count])
                    # print(data_dict['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg'))
                    image_paths[count] = data_dict['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')

        data_dict['image_paths'] = np.array(image_paths, dtype=object)
        # data_dict['pair_infos'] = np.asarray(data_dict['pair_infos'], dtype=object)

        new_file = os.path.join(
            './third_party/glue_factory_minima/data/megadepth/scene_info_no_sfm/', filename)
        np.savez(new_file, **data_dict)
        print("Saved to ", new_file)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Index Preparation")
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='./third_party/glue_factory_minima/data/megadepth/scene_info/',
                        help='Directory containing the scene info files')
    args = parser.parse_args()
    txt_path1 = "./third_party/glue_factory_minima/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean.txt"
    txt_path2 = "./third_party/glue_factory_minima/gluefactory/datasets/megadepth_scene_lists/valid_scenes_clean.txt"
    index_preparation(args.directory, txt_path1)
    index_preparation(args.directory, txt_path2)
