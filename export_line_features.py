"""
    Export line detections and descriptors given a list of input images.
"""
import os
import argparse
import cv2
import numpy as np
np.set_printoptions(threshold=1e100)
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from experiment import load_config
from model.line_matcher import LineMatcher
from misc.visualize_util import plot_images, plot_lines, plot_line_matches, plot_color_line_matches, plot_keypoints



def export_descriptors(images_list, ckpt_path, config, device, extension,
                       output_folder, multiscale=False):
    # Extract the image paths
    with open(images_list, 'r') as f:
        image_files = f.readlines()
        print(image_files)
    image_files = [path.strip('\n') for path in image_files]

    # Initialize the line matcher
    line_matcher = LineMatcher(
        config["model_cfg"], ckpt_path, device, config["line_detector_cfg"],
        config["line_matcher_cfg"], multiscale)
    print("\t Successfully initialized model")

    i = 0
    # Run the inference on each image and write the output on disk
    for img_path in tqdm(image_files):
        ###Change!!!

        img_path_root = os.path.join('/workspace/SOLD2-main/tea_room_images', img_path)#这句话是荣阁加上的
        img_raw = cv2.imread(img_path_root, 0)
        img = torch.tensor(img_raw[None, None] / 255., dtype=torch.float,
                           device=device)

        # Run the line detection and description
        ref_detection = line_matcher.line_detection(img)
        ref_line_seg = ref_detection["line_segments"]#检测出来的线段

        #Write the output on disk
        print(img_raw.shape)
        plot_images([img_raw], ['detected lines'])
        #fig_detect = plot_lines([ref_line_seg[:, :, ::-1]], ps=3, lw=2, indices=(0, 0))
        fig_detect = plot_lines([ref_line_seg[0:1, :, ::-1]], ps=3, lw=2, indices=(0, 0))
        print(ref_line_seg[0, :, ::-1])

        filename = os.path.splitext(os.path.basename(img_path))[0]
        savename = os.path.join(output_folder, filename + extension + '.jpg')
        fig_detect.savefig(savename)

        #存起来
        ref_line_seg = np.array([np.array(line).reshape(4, -1) for line in ref_line_seg])
        ref_line_seg = np.squeeze(ref_line_seg, 2)
        line_save_name = os.path.join(output_folder, "sold_line_detect%s.txt" % i)

        with open(line_save_name, "w") as f:
            f.write(str(ref_line_seg).replace('[', '').replace(']', ''))  # 自带文件关闭功能，不需要再写f.close()


        #ref_line_seg = np.array([line[:] for line in ref_line_seg])
        #print(ref_line_seg)
        print(ref_line_seg.shape, type(ref_line_seg))
        #a = ref_line_seg
        #a = a.astype(np.double)
        #print(a.shape, type(a))

        #a.astype('double').tofile(line_save_name)
        #np.savetxt(line_save_name, ref_line_seg)
        i = i+1

        #ref_descriptors = ref_detection["descriptor"][0].cpu().numpy()



        #np.savez_compressed(output_file, line_seg=ref_line_seg,
        #                    descriptors=ref_descriptors)


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_list", type=str, required=True,
                        help="List of input images in a text file.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder.")
    parser.add_argument("--config", type=str,
                        default="config/export_line_features.yaml")
    parser.add_argument("--checkpoint_path", type=str,
                        default="pretrained_models/sold2_wireframe.tar")
    parser.add_argument("--multiscale", action="store_true", default=False)
    parser.add_argument("--extension", type=str, default=None)
    args = parser.parse_args()

    # Get the device
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #device = torch.device("cpu")

    # Get the model config, extension and checkpoint path
    config = load_config(args.config)
    ckpt_path = os.path.abspath(args.checkpoint_path)
    extension = 'sold2' if args.extension is None else args.extension
    extension = "." + extension

    export_descriptors(args.img_list, ckpt_path, config, device, extension,
                       args.output_folder, args.multiscale)
