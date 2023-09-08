import math
import numpy as np
from operator import itemgetter

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
    heatmap_height, heatmap_width = heatmap_borders.shape[:2]

    heatmap_center = heatmap_borders[1:heatmap_height-1, 1:heatmap_width-1]
    heatmap_left = heatmap_borders[1:heatmap_height-1, 2:heatmap_width]
    heatmap_right = heatmap_borders[1:heatmap_height-1, 0:heatmap_width-2]
    heatmap_up = heatmap_borders[2:heatmap_height, 1:heatmap_width-1]
    heatmap_down = heatmap_borders[0:heatmap_height-2, 1:heatmap_width-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue

        for j in range(i+1, len(keypoints)):
            score = (keypoints[i][0] - keypoints[j][0])**2 + (keypoints[i][1] - keypoints[j][1])**2
            if math.sqrt(score) < 6:
                suppressed[j] = 1

        keypoint_with_score_and_id = (
            keypoints[i][0], 
            keypoints[i][1], 
            heatmap[keypoints[i][1], 
            keypoints[i][0]],
            total_keypoint_num + keypoint_num
        )
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1

    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def apply_nms(a_idx, b_idx, affinity_scores):
    # Apply non-maximum suppression to retrieved connections
    order = affinity_scores.argsort()[::-1]
    affinity_scores = affinity_scores[order]
    a_idx = a_idx[order]
    b_idx = b_idx[order]

    idx = []
    has_kpt_a, has_kpt_b = set(), set()
    for t, (i, j) in enumerate(zip(a_idx, b_idx)):
        if i not in has_kpt_a and j not in has_kpt_b:
            idx.append(t)
            has_kpt_a.add(i)
            has_kpt_b.add(j)

    idx = np.asarray(idx, dtype=np.int32)
    return a_idx[idx], b_idx[idx], affinity_scores[idx]


def group_keypoints(keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05):
    pose_entries = []
    all_keypoints = np.array([item for sublist in keypoints_by_type for item in sublist])
    points_per_limb = 10
    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
    keypoints_by_type = [np.array(kpt, np.float32) for kpt in keypoints_by_type]
    
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        n = len(kpts_a)
        m = len(kpts_b)
        if n == 0 or m == 0:
            continue

        # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
        a = kpts_a[:, :2]
        a = np.broadcast_to(a[None], (m, n, 2))
        b = kpts_b[:, :2]
        vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

        # Sample points along every candidate limb vector.
        steps = (1 / (points_per_limb - 1) * vec_raw)
        points = steps * grid + a.reshape(-1, 1, 2)
        points = points.round().astype(dtype=np.int32)
        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Compute affinity score between candidate limb vectors and part affinity field.
        field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
        vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
        vec = vec_raw / (vec_norm + 1e-6)
        affinity_scores = (field * vec).sum(-1).reshape(-1, points_per_limb)
        valid_affinity_scores = affinity_scores > min_paf_score
        valid_num = valid_affinity_scores.sum(1)
        affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
        success_ratio = valid_num / points_per_limb

        # Get a list of limbs according to the obtained affinity score.
        valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
        if len(valid_limbs) == 0:
            continue
        b_idx, a_idx = np.divmod(valid_limbs, n)
        affinity_scores = affinity_scores[valid_limbs]

        # Suppress incompatible connections.
        a_idx, b_idx, affinity_scores = apply_nms(a_idx, b_idx, affinity_scores)
        connections = list(zip(
            kpts_a[a_idx, 3].astype(np.int32),
            kpts_b[b_idx, 3].astype(np.int32),
            affinity_scores)
        )

        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                conn = connections[i]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = conn[0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = conn[1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[conn[0:2], 2]) + conn[2]
                
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == conn[0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = conn[1]
                    elif pose_entries[j][kpt_b_id] == conn[1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = conn[0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                conn = connections[i]
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == conn[0]:
                        pose_entries[j][kpt_b_id] = conn[1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[conn[1], 2] + conn[2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = conn[0]
                    pose_entry[kpt_b_id] = conn[1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[conn[0:2], 2]) + conn[2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for pose in pose_entries:
        if pose[-1] < 3 or (pose[-2] / pose[-1] < 0.2):
            continue
        filtered_entries.append(pose)

    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints
