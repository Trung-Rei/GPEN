import torch
from torch import nn
from basicsr.archs.stylegan2_arch import ConvLayer
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.ops import roi_align


@ARCH_REGISTRY.register()
class FacialComponentDiscriminator(nn.Module):
    """Facial component (eyes, mouth, noise) discriminator used in GFPGAN.
    """

    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()
        # It now uses a VGG-style architectrue with fixed model size
        self.conv1 = ConvLayer(3, 64, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv2 = ConvLayer(64, 128, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv3 = ConvLayer(128, 128, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv4 = ConvLayer(128, 256, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv5 = ConvLayer(256, 256, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False):
        """Forward function for FacialComponentDiscriminator.
        Args:
            x (Tensor): Input images.
            return_feats (bool): Whether to return intermediate features. Default: False.
        """
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)

        if return_feats:
            return out, rlt_feats
        else:
            return out, None


def get_roi_regions(gt, output, locs, eye_out_size=80, mouth_out_size=120, device="cpu"):
    face_ratio = int(gt.shape[-1] / 512)
    eye_out_size *= face_ratio
    mouth_out_size *= face_ratio

    loc_left_eyes, loc_right_eyes, loc_mouths = locs

    rois_eyes = []
    rois_mouths = []
    for b in range(loc_left_eyes.size(0)):  # loop for batch size
        # left eye and right eye
        img_inds = loc_left_eyes.new_full((2, 1), b)
        bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
        rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
        rois_eyes.append(rois)
        # mouse
        img_inds = loc_left_eyes.new_full((1, 1), b)
        rois = torch.cat([img_inds, loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        rois_mouths.append(rois)

    rois_eyes = torch.cat(rois_eyes, 0).to(device)
    rois_mouths = torch.cat(rois_mouths, 0).to(device)

    # real images
    all_eyes = roi_align(gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes_gt = all_eyes[0::2, :, :, :]
    right_eyes_gt = all_eyes[1::2, :, :, :]
    mouths_gt = roi_align(gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
    # output
    all_eyes = roi_align(output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes = all_eyes[0::2, :, :, :]
    right_eyes = all_eyes[1::2, :, :, :]
    mouths = roi_align(output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

    return left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes, right_eyes, mouths


def gram_mat(x):
    """Calculate Gram matrix.
    Args:
        x (torch.Tensor): Tensor with shape of (n, c, h, w).
    Returns:
        torch.Tensor: Gram matrix.
    """
    n, c, h, w = x.size()
    features = x.view(n, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


def comp_style(feat, feat_gt, criterion):
    return criterion(gram_mat(feat[0]), gram_mat(
        feat_gt[0].detach())) * 0.5 + criterion(
            gram_mat(feat[1]), gram_mat(feat_gt[1].detach()))