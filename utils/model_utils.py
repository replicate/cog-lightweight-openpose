import cv2
import numpy as np
import torch

from utils.keypoint_utils import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from utils.image_utils import OneEuroFilter, preprocess


def run_inference(net, img, net_input_height_size, stride, upsample_ratio, device):
    padded_img, pad, scale = preprocess(img, net_input_height_size)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = tensor_img.to(device)

    output = net(tensor_img)

    heatmaps = output[-2]
    heatmaps = np.transpose(heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(
        heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC
    )

    pafs = output[-1]
    pafs = np.transpose(pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(
        pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC
    )

    return heatmaps, pafs, scale, pad


class Pose:
    num_kpts = 18
    kpt_names = [
        'nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri', 'r_hip', 
        'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear'
    ]
    sigmas = [.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35]
    sigmas = np.array(sigmas, dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        found_kpts = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0

        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue

            found_kpts[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1

        bbox = cv2.boundingRect(found_kpts)
        return bbox

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)

            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]

            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)

            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)