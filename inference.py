# =====================================================
# File Name:    inference.py
# Project Name: Object Segmentation
# Description:  This program is used to inference image
#               or video.
#
# Usage:        $ python inference.py PATH_TO_IMAGE_OR_VIDEO
#
# Contributors:
# - Zechen Zhou     zzhou186@uottawa.ca
# - Shun Hei Yiu    syiu017@uottawa.ca
# =====================================================

import os
import time
import yaml
import torch
import numpy as np
import cv2
import argparse

from PIL import Image
from torchvision import transforms
from models import ICNet
from utils import SetupLogger, get_color_pallete

inference_output_path = "ckpt/inference_output"


class Inference(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_path = None
        self.video_path = None

        # get the inference input path
        input_path = _get_inference_input_path()
        if input_path.endswith('.png'):
            self.image_path = input_path
        elif input_path.endswith('.mp4'):
            self.video_path = input_path

        # create network
        self.model = ICNet(nclass=19, backbone='resnet50').to(self.device)

        # load ckpt
        pretrained_net = torch.load(
            cfg["test"]["ckpt_path"], map_location=self.device)
        self.model.load_state_dict(pretrained_net)
        self.model.eval()

    def start_inference(self):
        if self.image_path is not None:
            self._inference_image()
        elif self.video_path is not None:
            self._inference_video()

    def _inference_image(self):
        """ Runs inference on an image and saves the output """
        # image shape: (W,H,3)
        image = Image.open(self.image_path).convert('RGB')

        filename = os.path.basename(self.image_path)
        prefix = filename.split(".")[0]

        # Save original image
        image.save(os.path.join(outdir, prefix + '_src.png'))

        # Resize to (2048x1024) and save it
        image = image.resize((2048, 1024), Image.BILINEAR)
        image.save(os.path.join(outdir, prefix + '_src_resized.png'))

        image_tensor = self._img_transform(image)  # image shape: (3,H,W) [0,1]

        # image shape: (1,3,H,W) [0,1]
        image_tensor = image_tensor.to(self.device)

        image_tensor = torch.unsqueeze(image_tensor, 0)

        with torch.no_grad():
            inference_start = time.time()
            output = self.model(image_tensor)
            inference_end = time.time()

            inference_time = inference_end - inference_start

        logger.info("Sample: {:d}, inference time: {:.3f}s".format(
            1, inference_time))

        pred = torch.argmax(output[0], 1)
        pred = pred.cpu().data.numpy()
        pred = pred.squeeze(0)
        pred = get_color_pallete(pred, "citys")

        # Save inference result
        pred.save(os.path.join(outdir, f"{prefix}_pred.png"))
        print(f"Saved predicted image: {prefix}_pred.png")

    def _img_transform(self, image):
        """Preprocesses an image before passing it to the model."""
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        return image_transform(image)

    def _inference_video(self):
        """Runs inference on a video, frame by frame, and saves the output as a new video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Start inference, Total number of frames: {:d}".format(
            total_frames))
        list_preprocess_time = []
        list_inference_time = []
        list_postprocess_time = []

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        filename = os.path.basename(self.video_path)
        prefix = filename.split(".")[0]
        output_video_path = os.path.join(outdir, f"{prefix}_pred.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess Image
            preprocess_start = time.time()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self._img_transform(image).to(self.device)
            image_tensor = image_tensor.unsqueeze(0)  # (1,3,H,W)
            preprocess_end = time.time()
            preprocess_time = preprocess_end - preprocess_start
            list_preprocess_time.append(preprocess_time)

            # Inference
            inference_start = time.time()
            with torch.no_grad():
                output = self.model(image_tensor)
            inference_end = time.time()
            inference_time = inference_end - inference_start
            list_inference_time.append(inference_time)

            # Postprocess and write to video
            postprocess_start = time.time()
            pred = torch.argmax(output[0], 1).cpu().numpy().squeeze(0)
            pred = get_color_pallete(pred, "citys")
            pred = np.array(pred)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            out.write(pred)
            postprocess_end = time.time()
            postprocess_time = postprocess_end - postprocess_start
            list_postprocess_time.append(postprocess_time)

            frame_count += 1
            logger.info("Frame: {:d}, pre-process time: {:.3f}s, inference time: {:.3f}s, post-process time: {:.3f}s".format(
                frame_count, preprocess_time, inference_time, postprocess_time))

        cap.release()
        out.release()
        average_preprocess_time = sum(
            list_preprocess_time)/len(list_preprocess_time)
        average_inference_time = sum(
            list_inference_time)/len(list_inference_time)
        average_postprocess_time = sum(
            list_postprocess_time)/len(list_postprocess_time)
        logger.info("Average pre-process time: {:.3f}s, Average inference time: {:.3f}s, Average post-process time: {:.3f}s".format(
            average_preprocess_time, average_inference_time, average_postprocess_time))


def _get_inference_input_path():
    parser = argparse.ArgumentParser(
        description="Process command-line arguments.")

    # Add argument
    parser.add_argument(
        "--input", type=str, help="Path to the input image or video")

    # Parse arguments
    args = parser.parse_args()

    if args.input is None:
        print("Error: --input argument is required.")
        parser.print_help()
        exit(1)

    # Print argument
    print(f"Inference input file: {args.input}")

    return args.input


if __name__ == '__main__':
    # Set config file
    config_path = "./configs/icnet.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    # Use specific GPU
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))

    if torch.cuda.is_available():
        print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
        print("torch.cuda.current_device(): {}".format(
            torch.cuda.current_device()))

    outdir = os.path.join(cfg["train"]["ckpt_dir"], "inference_output")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["ckpt_dir"],
                         distributed_rank=0,
                         filename='{}_{}_evaluate_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))

    inference = Inference(cfg)
    inference.start_inference()
