import __init_paths
from training.data_loader.dataset_face import FaceDataset
from face_model.gpen_model import FullGenerator
import torch
import argparse
from torchvision import utils
import cv2
import numpy as np
from training.data_loader.dataset_face import GPEN_degradation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--pretrain', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--is_concat', action="store_true")

    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda"

    generator = FullGenerator(
        args.size, 512, 8, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device, isconcat=args.is_concat
    ).to(device)

    ckpt = torch.load(args.pretrain, map_location=device)
    generator.load_state_dict(ckpt["g_ema"])
    del ckpt
    torch.cuda.empty_cache()

    degrader = GPEN_degradation()

    img_gt = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img_gt = cv2.resize(img_gt, (args.size, args.size), interpolation=cv2.INTER_AREA)
    
    img_gt = img_gt.astype(np.float32)/255.
    img_gt, img_lq = degrader.degrade_process(img_gt)

    img_gt =  (torch.from_numpy(img_gt) - 0.5) / 0.5
    img_lq =  (torch.from_numpy(img_lq) - 0.5) / 0.5
    
    img_gt = img_gt.permute(2, 0, 1).flip(0) # BGR->RGB
    img_lq = img_lq.permute(2, 0, 1).flip(0) # BGR->RGB

    img = img_lq

    with torch.no_grad():
        out, __ = generator(img.reshape(1, 3, args.size, args.size).to(device))
        utils.save_image(out, args.output, normalize=True, range=(-1, 1))
    