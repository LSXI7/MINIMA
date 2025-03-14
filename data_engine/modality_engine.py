import argparse
import cv2
import numpy as np
import os
import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from tools.depth.depth_transfer import init_depth_model, depth_transfer_single
from tools.event.event_transfer import event_transfer_single
from tools.infrared.infrared_transfer import init_infrared_model, infrared_transfer_single
from tools.normal.normal_transfer import init_normal_model, normal_transfer_single
from tools.paint.paint_transfer import init_paint_model, paint_transfer_single
from tools.sketch.sketch_transfer import init_sketch_model, sketch_transfer_single


def transfer_infrared(args, input_paths, device, output_dir, ext):
    diff_infer, tuner_model_list = init_infrared_model(args, device)
    for image_path in tqdm(input_paths, desc="Processing Infrared Images"):
        images = infrared_transfer_single(args, image_path, diff_infer, tuner_model_list,
                                          big_enough_option=args.big_enough_option)
        count = 0
        for image in images:
            assert image.shape[0] == 3
            name = os.path.splitext(os.path.basename(image_path))[0]

            if count > 0:
                name = f"{name}_{count}"
            output_path = os.path.join(output_dir, f"{name}.{ext}")
            save_image(image, output_path)


def transfer_depth(args, input_paths, device, output_dir, ext):
    depth_model = init_depth_model(device, args.encoder)
    for image_path in tqdm(input_paths, desc="Processing Depth Images"):
        image = depth_transfer_single(image_path, depth_model)
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{name}.{ext}")
        cv2.imwrite(output_path, image)


def transfer_event(args, input_paths, output_dir, ext):
    contrast_min = args.c_min
    contrast_max = args.c_max
    shift_min = args.shift_min
    shift_max = args.shift_max
    for image_path in tqdm(input_paths, desc="Processing Event Images"):
        image = event_transfer_single(image_path, contrast_min, contrast_max, shift_min, shift_max)
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{name}.{ext}")
        cv2.imwrite(output_path, image)


def transfer_paint(args, input_paths, device, output_dir, ext):
    paint_model, patch_size, stroke_num = init_paint_model(args.model_path, device)
    if args.serial:
        serial = True
    else:
        serial = False
    for image_path in tqdm(input_paths, desc="Processing Paint Images"):
        image = paint_transfer_single(image_path, paint_model, device=device, serial=serial, patch_size=patch_size,
                                      stroke_num=stroke_num)
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{name}.{ext}")
        result = Image.fromarray((image.data.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
        result.save(output_path)


def transfer_normal(args, input_paths, device, output_dir, ext):
    model = init_normal_model(device)
    for image_path in tqdm(input_paths, desc="Processing Normal Images"):
        image = normal_transfer_single(image_path, model, device)
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{name}.{ext}")
        image.save(output_path)


def transfer_sketch(args, input_paths, device, output_dir, ext):
    model = init_sketch_model(args, device)
    load_size = args.load_size
    clahe_clip = args.clahe_clip
    for image_path in tqdm(input_paths, desc="Processing Sketch Images"):
        image = sketch_transfer_single(image_path, model, device, load_size, clahe_clip)
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{name}.{ext}")
        image.save(output_path)


if __name__ == "__main__":
    def add_common_arguments(parser):
        parser.add_argument('--input_path', type=str, default='./figs/origin_image.jpg', help='Path to input image or directory of images')
        parser.add_argument('--output_dir', type=str, default='./result', help='Directory for output images')
        parser.add_argument('--seed', type=int, default=-1, help='Random seed for noise')
        parser.add_argument('--ext', type=str, default='jpg', help='Output image extension')


    def add_method_arguments(parser, modality):
        if modality == "infrared":
            parser.add_argument('--style_guide_scale_text', type=float, default=7.5, help='Scale for style guide text')
            parser.add_argument('--style_guide_scale_image', type=float, default=1.5,
                                help='Scale for style guide image')
            parser.add_argument('--adapter_model_path', type=str, default='weights/stylebooth/step-210000/',
                                help='Path to adapter model')
            parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per image')
            parser.add_argument('--big_enough_option', type=int, default=1, help='Use big enough option')
        elif modality == "depth":
            parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                                help='Encoder to use for depth')
        elif modality == "event":
            parser.add_argument('--c_min', type=float, default=0.05, help='Minimum contrast value')
            parser.add_argument('--c_max', type=float, default=0.5, help='Maximum contrast value')
            parser.add_argument('--shift_min', type=int, default=1, help='Minimum shift value')
            parser.add_argument('--shift_max', type=int, default=3, help='Maximum shift value')
        elif modality == "paint":
            parser.add_argument('--model_path', type=str, default='weights/paint_transformer/model.pth',
                                help='Path to paint model')
            parser.add_argument('--serial', type=int, default=1, help='Use serial painting')
        elif modality == "normal":
            pass
        elif modality == "sketch":
            parser.add_argument('--model_variant', type=str, default='improved', choices=['default', 'improved'],
                                help='Model variant to use')
            parser.add_argument('--load_size', type=int, default=1024, help='Size to load images')
            parser.add_argument('--clahe_clip', type=float, default=10.0,
                                help='CLAHE clip value,-1 for no CLAHE')
        else:
            raise NotImplementedError(f"Modality {modality} not implemented")

        add_common_arguments(parser)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Modality Engine')
    parser.add_argument('--modality', type=str, required=True,
                        choices=['infrared', 'depth', 'normal', 'event', 'paint', 'sketch'],
                        help='Modalities to transfer')
    args, remaining_args = parser.parse_known_args()
    add_method_arguments(parser, args.modality)
    args = parser.parse_args()
    # print(args)

    input_paths = []
    if os.path.isdir(args.input_path):
        input_paths = [os.path.join(args.input_path, img) for img in os.listdir(args.input_path) if
                       img.endswith(('png', 'jpg', 'jpeg'))]
    elif os.path.isfile(args.input_path):
        input_paths = [args.input_path]

    output_dir = args.output_dir
    ext = args.ext
    os.makedirs(output_dir, exist_ok=True)

    if args.modality == "infrared":
        transfer_infrared(args, input_paths, device, output_dir, ext)
    elif args.modality == "depth":
        transfer_depth(args, input_paths, device, output_dir, ext)
    elif args.modality == "event":
        transfer_event(args, input_paths, output_dir, ext)
    elif args.modality == "paint":
        transfer_paint(args, input_paths, device, output_dir, ext)
    elif args.modality == "normal":
        transfer_normal(args, input_paths, device, output_dir, ext)
    elif args.modality == "sketch":
        transfer_sketch(args, input_paths, device, output_dir, ext)
    else:
        raise NotImplementedError(f"Modality {args.modality} not implemented")
