# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from typing import Any, List, Dict, Optional

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.mobilenet import PoseEstimationWithMobileNet, load_state
from utils.keypoint_utils import extract_keypoints, group_keypoints
from utils.model_utils import Pose, run_inference
from utils.image_utils import preprocess
from cog import BasePredictor, BaseModel, Input, Path


class ModelOutput(BaseModel):
    keypoint_names: List
    json_data: Dict
    keypoints_img: Optional[Path]

    
class Predictor(BasePredictor):
    def setup(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load("./checkpoints/checkpoint_iter_370000.pth", map_location=device)
        self.model = PoseEstimationWithMobileNet()
        load_state(self.model, checkpoint)

        self.device = torch.device(device)
        self.model.eval()
        self.model.to(device)

        self.stride = 8
        self.upsample_ratio = 4

    def predict(
        self,
        image: Path = Input(description="RGB input image"),
        image_size: int = Input(description="Image size for inference", ge=128, le=1024, default=256),
        show_visualisation: bool = Input(
            description="Draw and visualize keypoints on the image", default=False
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        keypoint_names = [
            "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", 
            "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", 
            "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"
        ]
        json_data = {"objects": [],}  # id, confidence, bbox, keypoints
        num_keypoints = Pose.num_kpts

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        orig_img = img.copy()

        heatmaps, pafs, scale, pad = run_inference(
            self.model, img, image_size, self.stride, self.upsample_ratio, self.device
        )

        total_keypoints_num = 0
        all_kpts_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_kpts_by_type, total_keypoints_num)

        pose_entries, all_kpts = group_keypoints(all_kpts_by_type, pafs)
        for kpt_id in range(all_kpts.shape[0]):
            all_kpts[kpt_id, 0] = (all_kpts[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_kpts[kpt_id, 1] = (all_kpts[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale

        # Iterate over pose of each person
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue

            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                # If keypoint is found
                if pose_entries[n][kpt_id] != -1.0:  
                    pose_keypoints[kpt_id, 0] = int(all_kpts[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_kpts[int(pose_entries[n][kpt_id]), 1])

            pose = Pose(pose_keypoints, pose_entries[n][18])
            data = {
                "id": n, 
                "confidence": pose.confidence,
                "bbox": list(pose.bbox),
                "keypoint_coords": pose.keypoints.tolist(),
            }
            json_data["objects"].append(data)
            current_poses.append(pose)


        if show_visualisation:
            for pose in current_poses:
                pose.draw(img)

            img = cv2.addWeighted(orig_img, 0.4, img, 0.6, 0)
            keypoints_img_path = "/tmp/keypoints.png"
            #cv2.imwrite(keypoints_img_path, img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis("off")
            
            plt.savefig(keypoints_img_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
            plt.close()

        return ModelOutput(
            keypoint_names=keypoint_names,
            json_data=json_data,
            keypoints_img=Path(keypoints_img_path) if show_visualisation else None,
        )