# ========================================================================
# ORION INFERENCE DEMO V78 (FIX: FLATTENED MEDIAN + GROUND SHIFT)
# ========================================================================
# 1. FIXED (Cam): Lanes are now Flattened (Median Z) AND Shifted (-1.85m)
#    to fix the "hovering" issue seen in V77.
# 2. RETAINED (Cam): Boxes use raw model output (no shift) as they are correct.
# 3. RETAINED (BEV): Lateral Inversion (cx + x) to correct left/right flip.
# ========================================================================

import cv2
cv2.setNumThreads(0)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["cv_num_threads"] = "1"

# ========================================================================
# IMPORTS & GLOBAL PATCH
# ========================================================================
import torch
import numpy as np

# CRITICAL PATCH: Restore np.int for any other legacy library calls
if not hasattr(np, 'int'):
    np.int = int

import imageio
import gc
import sys
import types
import re
import copy
from tqdm import tqdm
from math import factorial, cos, sin
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from mmcv.datasets import build_dataset
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Constants ---
LIDAR_HEIGHT_CORRECTION = 1.85 

# --- ORION Placeholder Class ---
class ORION:
    pass

# --- 1. LIBRARY STRUCTURES (Use Library as requested) ---
try:
    from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
except ImportError:
    print("Warning: Library LiDARInstance3DBoxes not found. Using local fallback.")
    class LiDARInstance3DBoxes:
        def __init__(self, tensor, box_dim=7, origin=(0.5, 0.5, 0)):
            self.tensor = tensor
            self.box_dim = box_dim
            self.origin = origin

        @property
        def corners(self):
            dims = self.tensor[:, 3:6]
            rot = self.tensor[:, 6]
            origin = self.tensor.new_tensor(self.origin)
            x_corners = [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5]
            y_corners = [0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5]
            z_corners = [0, 0, 0, 0, 1, 1, 1, 1]
            corners_norm = torch.tensor(np.stack([x_corners, y_corners, z_corners], axis=1), dtype=self.tensor.dtype, device=self.tensor.device)
            corners_norm = corners_norm - origin
            corners = dims.view([-1, 1, 3]) * corners_norm.view([1, 8, 3])
            c = torch.cos(rot)
            s = torch.sin(rot)
            rot_mat = torch.zeros((len(rot), 3, 3), device=self.tensor.device)
            rot_mat[:, 0, 0] = c
            rot_mat[:, 0, 1] = -s
            rot_mat[:, 1, 0] = s
            rot_mat[:, 1, 1] = c
            rot_mat[:, 2, 2] = 1
            corners = torch.bmm(corners, rot_mat.transpose(1, 2))
            corners += self.tensor[:, :3].view(-1, 1, 3)
            return corners

# --- 2. LOCAL VISUALIZATION (Overrides Library to fix np.int crash) ---
def points_cam2img(points_3d, proj_mat, with_depth=False):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    d1, d2 = proj_mat.shape[:2]
    if d1 == 3:
        proj_mat_expanded = np.eye(4, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded
        
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    
    if with_depth:
        return np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)
    return point_2d_res

def plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color=(0, 255, 0), thickness=1):
    img = img.copy()
    for i in range(num_bbox):
        pts = imgfov_pts_2d[i]
        pts = np.clip(pts, -10000, 10000) 
        
        # CRITICAL FIX: Use 'int' instead of 'np.int'
        corners = pts.astype(int)
        
        cv2.line(img, tuple(corners[4]), tuple(corners[5]), color, thickness)
        cv2.line(img, tuple(corners[5]), tuple(corners[6]), color, thickness)
        cv2.line(img, tuple(corners[6]), tuple(corners[7]), color, thickness)
        cv2.line(img, tuple(corners[7]), tuple(corners[4]), color, thickness)
        cv2.line(img, tuple(corners[0]), tuple(corners[1]), color, thickness)
        cv2.line(img, tuple(corners[1]), tuple(corners[2]), color, thickness)
        cv2.line(img, tuple(corners[2]), tuple(corners[3]), color, thickness)
        cv2.line(img, tuple(corners[3]), tuple(corners[0]), color, thickness)
        cv2.line(img, tuple(corners[0]), tuple(corners[4]), color, thickness)
        cv2.line(img, tuple(corners[1]), tuple(corners[5]), color, thickness)
        cv2.line(img, tuple(corners[2]), tuple(corners[6]), color, thickness)
        cv2.line(img, tuple(corners[3]), tuple(corners[7]), color, thickness)
    return img

def draw_lidar_bbox3d_on_img(bboxes3d, raw_img, lidar2img_rt, img_metas, color=(0, 255, 0), thickness=1):
    # Works with both library and local box objects
    if hasattr(bboxes3d, 'tensor'):
        corners_3d = bboxes3d.corners.detach().cpu().numpy()
    else:
        corners_3d = bboxes3d.corners
        
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    
    pts_2d = pts_4d @ lidar2img_rt.T
    
    # Clip depth to prevent division by zero or negative depth artifacts
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=0.1, a_max=1e5)
    
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
    return plot_rect3d_on_img(raw_img, num_bbox, imgfov_pts_2d, color, thickness)

# ========================================================================
# HELPER: MATH & GEOMETRY
# ========================================================================
def to_cv_pt(point):
    x = point[0]
    y = point[1]
    x = max(min(x, 50000), -50000)
    y = max(min(y, 50000), -50000)
    return (int(x), int(y))

def bezier_interpolation(controls, n_points=50):
    if len(controls) == 0:
        return np.zeros((0, n_points, 3))
    n_control = controls.shape[1]
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    for i in range(n_points):
        for j in range(n_control):
            comb = factorial(n_control - 1) // (factorial(j) * factorial(n_control - 1 - j))
            basis = comb * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
            A[i, j] = basis
    return np.einsum('ij,njk->nik', A, controls)

# ========================================================================
# 1. VISUALIZATION ENGINE
# ========================================================================
def render_cam_frame(real_data, trajectory, bbox_results, lane_results, frame_idx, total):
    img_t = real_data['img']
    if img_t.dim() == 5: img_t = img_t[0, 0]
    elif img_t.dim() == 4: img_t = img_t[0]
    img = img_t.detach().cpu().float().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(np.transpose(img, (1, 2, 0)))
    H_img, W_img = img.shape[:2]
    
    l2i = real_data['lidar2img']
    if l2i.dim() == 4: l2i = l2i[0, 0]
    elif l2i.dim() == 3: l2i = l2i[0]
    l2i_np = l2i.detach().cpu().float().numpy()

    # --- 1. LANES ---
    if lane_results is not None and len(lane_results) > 0:
        try:
            detection = lane_results[0]
            if 'map_pts_3d' in detection:
                controls = detection['map_pts_3d'].detach().cpu().numpy()
                scores = detection['map_scores_3d'].detach().cpu().numpy()
                dense_lanes = bezier_interpolation(controls, n_points=40)
                
                for i, lane_pts_3d in enumerate(dense_lanes):
                    if scores[i] > 0.35: 
                        lane_lidar = lane_pts_3d.copy()
                        
                        # --- FIX 1: FLATTEN + GROUND SHIFT ---
                        # 1. Flatten: Snap all points to the median Z (removes waviness)
                        median_z = np.median(lane_lidar[:, 2])
                        lane_lidar[:, 2] = median_z
                        
                        # 2. Shift: If median is hovering (near 0), drop it to ground (-1.85)
                        # We force the drop here because V77 confirmed they were hovering.
                        lane_lidar[:, 2] -= LIDAR_HEIGHT_CORRECTION
                        
                        pts_2d_depth = points_cam2img(lane_lidar, l2i_np, with_depth=True)
                        mask = pts_2d_depth[:, 2] > 0.1
                        valid_pts = pts_2d_depth[mask][:, :2]
                        if len(valid_pts) > 1:
                            for k in range(len(valid_pts) - 1):
                                pt1 = to_cv_pt(valid_pts[k])
                                pt2 = to_cv_pt(valid_pts[k+1])
                                if (0 <= pt1[0] < W_img * 1.5) and (0 <= pt1[1] < H_img * 1.5):
                                    cv2.line(img, pt1, pt2, (0, 255, 127), 3, cv2.LINE_AA)
        except Exception as e:
            pass

    # --- 2. TRAJECTORY ---
    if trajectory is not None and trajectory.dim() == 2 and trajectory.shape[0] > 0:
        if torch.max(torch.abs(trajectory)) > 0.01:
            traj_np = trajectory.detach().cpu().float().numpy()
            z_vals = np.full((traj_np.shape[0], 1), -LIDAR_HEIGHT_CORRECTION)
            traj_3d = np.hstack((traj_np, z_vals))
            
            pts_2d_depth = points_cam2img(traj_3d, l2i_np, with_depth=True)
            mask = pts_2d_depth[:, 2] > 0.1
            valid_pts = pts_2d_depth[mask][:, :2]
            
            for k in range(len(valid_pts) - 1):
                pt1 = to_cv_pt(valid_pts[k])
                pt2 = to_cv_pt(valid_pts[k+1])
                cv2.line(img, pt1, pt2, (255, 50, 50), 3, cv2.LINE_AA)
            for k in range(len(valid_pts)):
                pt = to_cv_pt(valid_pts[k])
                cv2.circle(img, pt, 5, (255, 100, 100), -1)

    # --- 3. BOUNDING BOXES ---
    if bbox_results is not None and len(bbox_results) > 0:
        try:
            bbox_obj = bbox_results[0][0]
            if hasattr(bbox_obj, 'tensor'): bboxes_3d_tensor = bbox_obj.tensor
            elif hasattr(bbox_obj, 'data'): bboxes_3d_tensor = bbox_obj.data
            elif isinstance(bbox_obj, torch.Tensor): bboxes_3d_tensor = bbox_obj
            else: bboxes_3d_tensor = torch.tensor(bbox_obj)
            
            scores = bbox_results[0][1]
            if isinstance(scores, torch.Tensor): scores = scores.detach().cpu().numpy()
            
            # Low threshold to catch car in rain
            mask = scores > 0.1 
            
            if mask.any():
                # Explicit CPU move
                valid_boxes = bboxes_3d_tensor[mask].clone().detach().cpu()
                
                # --- FIX 2: TRUST MODEL FOR BOXES ---
                # User confirmed boxes are correct without Z-shift.
                # valid_boxes[:, 2] -= LIDAR_HEIGHT_CORRECTION  <-- REMOVED
                
                if valid_boxes.dim() == 1:
                    valid_boxes = valid_boxes.unsqueeze(0)
                
                # Use Library Structure (if available)
                lidar_boxes = LiDARInstance3DBoxes(
                    valid_boxes, 
                    box_dim=valid_boxes.shape[-1], 
                    origin=(0.5, 0.5, 0)
                )
                
                # USE LOCAL DRAW FUNCTION (Fixes crash)
                img = draw_lidar_bbox3d_on_img(lidar_boxes, img, l2i_np, None, color=(0, 165, 255), thickness=2)
        except Exception as e:
            print(f"[Warn] Box rendering skip: {e}")
            pass

    text = f'Frame {frame_idx+1}/{total}'
    cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def render_bev_frame(trajectories, lane_results, bbox_results, current_idx):
    H, W = 800, 800
    bev_img = np.ones((H, W, 3), dtype=np.uint8) * 30
    scale = W / 100.0
    cx, cy = W // 2, H // 2 + 200

    def world2pix(x, y):
        # BEV Mapping: X=Lateral, Y=Longitudinal (in this specific layout)
        # --- FIX 3: LATERAL INVERSION (RETAINED) ---
        # Changed 'cx - x' to 'cx + x' to flip Left/Right
        # without changing the vertical axis.
        u = int(cx + x * scale)
        v = int(cy - y * scale)
        return (u, v)

    # Grid
    for i in range(-20, 120, 20):
        cv2.line(bev_img, world2pix(-50, i), world2pix(50, i), (60, 60, 60), 1)
    for i in range(-50, 50, 10):
        cv2.line(bev_img, world2pix(i, 0), world2pix(i, 100), (60, 60, 60), 1)

    # 1. Lanes
    if lane_results is not None and len(lane_results) > 0:
        try:
            detection = lane_results[0]
            if 'map_pts_3d' in detection:
                controls = detection['map_pts_3d'].detach().cpu().numpy()
                scores = detection['map_scores_3d'].detach().cpu().numpy()
                dense_lanes = bezier_interpolation(controls, n_points=50)
                
                for i, lane_pts_3d in enumerate(dense_lanes):
                    if scores[i] > 0.35:
                        pts_pix = np.array([world2pix(pt[0], pt[1]) for pt in lane_pts_3d], dtype=np.int32)
                        cv2.polylines(bev_img, [pts_pix.reshape((-1, 1, 2))], False, (100, 255, 100), 2)
        except:
            pass

    # 2. Bounding Boxes
    if bbox_results is not None and len(bbox_results) > 0:
        try:
            bbox_obj = bbox_results[0][0]
            if hasattr(bbox_obj, 'tensor'): bboxes_3d = bbox_obj.tensor
            elif isinstance(bbox_obj, torch.Tensor): bboxes_3d = bbox_obj
            else: bboxes_3d = torch.tensor(bbox_obj)
            
            scores = bbox_results[0][1]
            if isinstance(scores, torch.Tensor): scores = scores.detach().cpu().numpy()
            labels = bbox_results[0][2]
            if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
            
            mask = scores > 0.1
            if mask.any():
                valid_boxes = bboxes_3d[mask]
                valid_labels = labels[mask]
                lidar_boxes = LiDARInstance3DBoxes(valid_boxes, box_dim=valid_boxes.shape[-1], origin=(0.5, 0.5, 0))
                bev_corners = lidar_boxes.corners[:, [0, 3, 7, 4], :2].detach().cpu().numpy()
                
                for k, box_corners in enumerate(bev_corners):
                    poly_pts = np.array([world2pix(c[0], c[1]) for c in box_corners], dtype=np.int32)
                    color = (0, 165, 255)
                    if valid_labels[k] >= 2: color = (255, 255, 0)
                    cv2.polylines(bev_img, [poly_pts.reshape((-1, 1, 2))], True, color, 2)
        except Exception as e:
            pass

    # 3. Trajectory
    c = trajectories[current_idx]
    if c.dim() == 2 and torch.max(torch.abs(c)) > 0.01:
        c_np = c.detach().cpu().float().numpy()
        pts_pix = [world2pix(p[0], p[1]) for p in c_np]
        for j in range(len(pts_pix)-1):
            cv2.line(bev_img, pts_pix[j], pts_pix[j+1], (0, 0, 255), 3)

    # 4. Ego Car
    ego_u, ego_v = world2pix(0, 0)
    pts_car = np.array([
        [ego_u, ego_v - 15],        # Tip (Up)
        [ego_u + 10, ego_v + 10],  # Rear Right
        [ego_u, ego_v + 5],        # Indentation
        [ego_u - 10, ego_v + 10]   # Rear Left
    ], dtype=np.int32)
    cv2.fillPoly(bev_img, [pts_car], (255, 0, 0))
    
    return bev_img

# ========================================================================
# 2. MODEL PATCHES (STABLE)
# ========================================================================
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

def safe_prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes=None):
    if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
        
    if image_features is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

    if isinstance(image_features, list):
        temp_image_features = []
        for b_id in range(len(image_features[0])):
            for img_id in range(len(image_features)):
                temp_image_features.append(image_features[img_id][b_id])
        image_features = torch.stack(temp_image_features).to(dtype=self.base_model.dtype)
    else:
        image_features = image_features.reshape(image_features.shape[0], -1, self.base_model.config.hidden_size).to(dtype=self.base_model.dtype)

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()

    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_list = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels_list = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    new_input_ids = []
    cur_image_idx = 0
    vocab_size = self.get_model().embed_tokens.num_embeddings

    for batch_idx, cur_input_ids in enumerate(input_ids_list):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        clean_input_ids = cur_input_ids.clone()
        clean_input_ids[clean_input_ids < 0] = 0
        clean_input_ids[clean_input_ids >= vocab_size] = 0

        if num_images == 0:
            if cur_image_idx < image_features.shape[0]:
                cur_image_features = image_features[cur_image_idx]
            else:
                cur_image_features = torch.empty((0, self.base_model.config.hidden_size), device=self.base_model.device)
            cur_input_embeds_1 = self.get_model().embed_tokens(clean_input_ids.to(self.base_model.device))
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels_list[batch_idx])
            new_input_ids.append(cur_input_ids)
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels_list[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        split_sizes = [x.shape[0] for x in cur_input_ids_noim]
        
        if len(cur_input_ids_noim) > 0:
            full_input_ids_noim = torch.cat(cur_input_ids_noim)
            clean_full_input_ids = full_input_ids_noim.clone()
            clean_full_input_ids[clean_full_input_ids < 0] = 0
            clean_full_input_ids[clean_full_input_ids >= vocab_size] = 0
            cur_input_embeds = self.get_model().embed_tokens(clean_full_input_ids.to(self.base_model.device))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        else:
            cur_input_embeds_no_im = []

        cur_new_input_embeds = []
        cur_new_labels = []
        cur_new_input_ids = []

        for i in range(num_images + 1):
            if i < len(cur_input_embeds_no_im):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_input_ids.append(cur_input_ids_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
        
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)
        cur_new_input_ids = torch.cat(cur_new_input_ids)
        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        new_input_ids.append(cur_input_ids)

    if len(new_input_embeds) == 0:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)
    ref_tensor = new_input_ids[0]
    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    new_inputs_ids_padded = torch.zeros((batch_size, max_len), dtype=ref_tensor.dtype, device=ref_tensor.device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels, cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
        cur_len = cur_new_embed.shape[0]
        ids_len = cur_new_input_ids.shape[0]
        safe_len = min(cur_len, ids_len)
        new_input_embeds_padded.append(torch.cat((
            cur_new_embed[:safe_len],
            torch.zeros((max_len - safe_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
        ), dim=0))
        if safe_len > 0:
            new_labels_padded[i, :safe_len] = cur_new_labels[:safe_len]
            new_inputs_ids_padded[i, :safe_len] = cur_new_input_ids[:safe_len]
            attention_mask[i, :safe_len] = True
            position_ids[i, :safe_len] = torch.arange(0, safe_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None: new_labels = None
    else: new_labels = new_labels_padded
    if _attention_mask is None: attention_mask = None
    else: attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
    if _position_ids is None: position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_inputs_ids_padded

def apply_safe_multimodal_patch(model):
    print("\n[Hard Patch] Applying Multimodal Safe Patch...")
    def recursive_apply(module):
        if hasattr(module, 'prepare_inputs_labels_for_multimodal'):
            module.prepare_inputs_labels_for_multimodal = types.MethodType(safe_prepare_inputs_labels_for_multimodal, module)
        for child in module.children():
            recursive_apply(child)
    recursive_apply(model)

def safe_simple_test_pts(self, img_metas, **data):
    import re
    import torch
    import numpy as np
    try:
        device = next(self.parameters()).device
    except:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if self.with_pts_bbox:
        if 'img' in data:
            x = self.extract_feat(data['img'])
        else:
            x = self.extract_feat(next(iter(data.values())))
        data['img_feats'] = x
        location = self.prepare_location(img_metas, **data)
        self.forward_roi_head(location, **data)
        pos_embed = self.position_embeding(data, location, img_metas)
        
        bbox_results = []
        pts_outs = {}
        vision_embeded_obj = None

        if self.with_pts_bbox:
            outs, det_query = self.pts_bbox_head(img_metas, pos_embed, **data)
            vision_embeded_obj = det_query.clone()
            if hasattr(self.pts_bbox_head, 'forward_test'):
                pts_outs.update(self.pts_bbox_head.forward_test(x, img_metas, **data)['pts_bbox'])
            else:
                try:
                    if self.use_col_loss:
                        bbox_list_raw = self.pts_bbox_head.get_motion_bboxes(outs, img_metas)
                    else:
                        bbox_list_raw = self.pts_bbox_head.get_bboxes(outs, img_metas)
                    if bbox_list_raw:
                        bbox_results.append(bbox_list_raw[0])
                except:
                    pass
        
        if isinstance(outs, dict): pts_outs.update(outs)
        
        lane_results = None
        vision_embeded_map = None
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            lane_results = self.map_head.get_bboxes(outs, img_metas)

        metric_dict = pts_outs.get('metric_dict', {})
        generated_text = []
        ego_fut_preds = torch.zeros((6, 2), device=device)

        if self.with_lm_head and vision_embeded_obj is not None and vision_embeded_map is not None:
            history_input_output_id = []
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
            input_ids_list = data.get('input_ids', [[]])[0]
            
            for i, input_ids in enumerate(input_ids_list):
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() == 0: input_ids = input_ids.unsqueeze(0).unsqueeze(0)
                    elif input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
                
                if input_ids.shape[-1] <= 1:
                    prompt_text = "Please provide the planning trajectory for the ego car without reasons."
                    tokenizer = getattr(self, 'tokenizer', None)
                    if tokenizer:
                        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
                
                history_input_output_id.append(input_ids)
            
            context_input_ids = torch.cat(history_input_output_id,dim=-1)
            output_ids = self.lm_head.generate(
                inputs=context_input_ids,
                images=vision_embeded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=320,
                use_cache=True,
                repetition_penalty=2.0
            )
            text_out = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            safe_Q = "VLM Label Missing (Inference Mode)"
            vlm_labels_data = img_metas[0].get('vlm_labels')
            if vlm_labels_data is not None and hasattr(vlm_labels_data, 'data') and len(vlm_labels_data.data) > i:
                safe_Q = vlm_labels_data.data[i]
                
            generated_text.append(dict(Q=safe_Q, A=text_out))

        if len(generated_text) > 0:
            traj = generated_text[0]['A'][0]
            full_match = re.search(r'\[PT,\s*\(\s*([+\-]?\d*\.?\d+),\s*([+\-]?\d*\.?\d+)\s*\)(?:,\s*\(\s*([+\-]?\d*\.?\d+),\s*([+\-]?\d*\.?\d+)\s*\))*\]', traj)
            if full_match:
                coords_iter = re.findall(r'\(\s*([+\-]?\d*\.?\d+),\s*([+\-]?\d*\.?\d+)\s*\)', traj)
                if coords_iter:
                    coordinates = [tuple(map(float, coord)) for coord in coords_iter]
                    ego_fut_preds = torch.tensor(np.array(coordinates)).to(device)
                    if len(ego_fut_preds) != 6:
                        ego_fut_preds = torch.zeros((6,2), device=device)

        if ego_fut_preds.shape == (6, 2):
            if torch.max(torch.abs(ego_fut_preds)) < 10.0:
                ego_fut_preds = ego_fut_preds.cumsum(dim=0)

        if lane_results and len(lane_results) > 0:
            lane_results[0]['ego_fut_preds'] = ego_fut_preds.float()
            lane_results[0]['ego_fut_cmd'] = data.get('ego_fut_cmd', None)
            lane_results[0]['fut_valid_flag'] = True
        else:
            if lane_results is None: lane_results = [{}]
            lane_results[0]['ego_fut_preds'] = ego_fut_preds.float()
            lane_results[0]['ego_fut_cmd'] = data.get('ego_fut_cmd', None)
            lane_results[0]['fut_valid_flag'] = True
            
        return bbox_results, generated_text, lane_results, metric_dict

def apply_safe_test_pts_patch(model):
    def patch(m):
        m.simple_test_pts = types.MethodType(safe_simple_test_pts, m)
    if isinstance(model, ORION): patch(model)
    elif hasattr(model, 'module') and isinstance(model.module, ORION): patch(model.module)
    else: patch(model)

def safe_simple_test(self, img_metas, **data):
    bbox_list, generated_text, lane_results, metric_dict = self.simple_test_pts(img_metas, **data)
    if bbox_list is None: bbox_list = [[]] * len(img_metas)
    return [{'pts_bbox': {'bbox_pts': bbox_list, 'lane_results': lane_results, 'generated_text': generated_text}}]

def apply_safe_test_patch(model):
    apply_safe_test_pts_patch(model)
    def patch(m):
        m.simple_test = types.MethodType(safe_simple_test, m)
    if isinstance(model, ORION): patch(model)
    elif hasattr(model, 'module') and isinstance(model.module, ORION): patch(model.module)
    else: patch(model)

# ========================================================================
# 3. RUN LOOP
# ========================================================================
def aggressive_unwrap(data):
    if hasattr(data, 'data') and not isinstance(data, (torch.Tensor, np.ndarray, str)):
        return aggressive_unwrap(data.data)
    if isinstance(data, list): return [aggressive_unwrap(x) for x in data]
    if isinstance(data, tuple): return tuple(aggressive_unwrap(x) for x in data)
    return data

def force_tensor(data):
    if isinstance(data, torch.Tensor): return data
    if isinstance(data, np.ndarray): return torch.from_numpy(data)
    if isinstance(data, (int, float)): return torch.tensor([data])
    if isinstance(data, list):
        if len(data) == 0: return torch.tensor([])
        converted = [force_tensor(x) for x in data]
        if len(converted) > 0 and isinstance(converted[0], torch.Tensor):
            try: return torch.stack(converted)
            except: return converted[0]
    return data

def process_to_gpu_batch(obj, name="unknown"):
    obj = aggressive_unwrap(obj)
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    obj = force_tensor(obj)
    if not isinstance(obj, torch.Tensor): return obj
    
    if name == 'img' and obj.dim() == 4:
        obj = obj.unsqueeze(0)
    elif name in ['lidar2img', 'cam_intrinsic'] and obj.dim() == 3:
        obj = obj.unsqueeze(0)
    elif name in ['ego_pose', 'ego_pose_inv'] and obj.dim() == 2:
        obj = obj.unsqueeze(0)
    elif name in ['input_ids', 'ego_fut_cmd']:
        if obj.dim() == 0: obj = obj.unsqueeze(0).unsqueeze(0)
        elif obj.dim() == 1: obj = obj.unsqueeze(0)
    elif name == 'command':
        if obj.dim() == 0: obj = obj.unsqueeze(0)
    if name == 'can_bus' and obj.dim() == 1:
        obj = obj.unsqueeze(0)
        
    if name == 'input_ids': return obj.cuda(non_blocking=False).long()
    return obj.cuda(non_blocking=False).float()

def extract_field(data_batch, key):
    if key not in data_batch: return None
    return data_batch[key]

def force_disable_flash_attn(cfg):
    print("\n[Config Patch] Disabling Flash Attention...")
    if 'img_backbone' in cfg.model: cfg.model.img_backbone.flash_attn = False
    if 'map_head' in cfg.model and 'transformer' in cfg.model.map_head: cfg.model.map_head.transformer.flash_attn = False
    if 'pts_bbox_head' in cfg.model:
        if 'transformer' in cfg.model.pts_bbox_head: cfg.model.pts_bbox_head.transformer.flash_attn = False
    return cfg

def run_safe_inference_stream(dataset, indices, model):
    print(f"\nInitializing Direct-to-Disk GIF writers...")
    cam_writer = imageio.get_writer('stream_orion_cam.gif', mode='I', fps=2, loop=0)
    bev_writer = imageio.get_writer('stream_orion_bev.gif', mode='I', fps=2, loop=0)
    traj_history = []
    
    print(f"Starting Stream Processing ({len(indices)} frames)...")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    for i, idx in enumerate(tqdm(indices, desc="Processing", file=sys.stdout)):
        torch.cuda.empty_cache()
        gc.collect()
        try:
            data_info = dataset.get_data_info(idx)
            input_dict = data_info.copy()
            dataset.pre_pipeline(input_dict)
            example = dataset.pipeline(input_dict)
            
            real_data = {}
            for k, val in example.items():
                if k == 'img_metas': continue
                if val is not None: real_data[k] = process_to_gpu_batch(val, k)
            
            if 'input_ids' not in real_data or real_data['input_ids'] is None:
                prompt_text = "Please provide the planning trajectory for the ego car without reasons."
                tokenizer = getattr(model, 'tokenizer', getattr(getattr(model, 'module', None), 'tokenizer', None))
                if tokenizer:
                    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                    real_data['input_ids'] = torch.tensor([tokens], dtype=torch.long).cuda()
                else:
                    real_data['input_ids'] = torch.tensor([[1]], dtype=torch.long).cuda()
            else:
                if real_data['input_ids'].numel() <= 1:
                    prompt_text = "Please provide the planning trajectory for the ego car without reasons."
                    tokenizer = getattr(model, 'tokenizer', getattr(getattr(model, 'module', None), 'tokenizer', None))
                    if tokenizer:
                        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                        real_data['input_ids'] = torch.tensor([tokens], dtype=torch.long).cuda()
                elif real_data['input_ids'].dim() == 1:
                    real_data['input_ids'] = real_data['input_ids'].unsqueeze(0)
            real_data['input_ids'] = real_data['input_ids'].cuda().long()

            img_metas_raw = extract_field(example, 'img_metas')
            img_metas = aggressive_unwrap(img_metas_raw)
            if isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], list):
                img_metas = img_metas[0]
            if isinstance(img_metas, dict): img_metas = [img_metas]

            with torch.no_grad():
                result = model.simple_test(img_metas, **real_data)
                
            pred_traj = torch.zeros(6, 2)
            bbox_results = []
            lane_results = None
            
            if isinstance(result, list) and len(result) > 0:
                res = result[0]
                if 'pts_bbox' in res:
                    pts_data = res['pts_bbox']
                    if 'lane_results' in pts_data and len(pts_data['lane_results']) > 0:
                        lane_res = pts_data['lane_results'][0]
                        if 'ego_fut_preds' in lane_res:
                            pred_traj = lane_res['ego_fut_preds'].detach().cpu().float()
                    if 'bbox_pts' in pts_data: bbox_results = pts_data['bbox_pts']
                    if 'lane_results' in pts_data: lane_results = pts_data['lane_results']
            
            traj_history.append(pred_traj)
            
            cam_frame = render_cam_frame(real_data, pred_traj, bbox_results, lane_results, i, len(indices))
            bev_frame = render_bev_frame(traj_history, lane_results, bbox_results, i)
            
            cam_writer.append_data(cam_frame)
            bev_writer.append_data(bev_frame)
            del real_data, example, input_dict, result, cam_frame, bev_frame
            
        except Exception as e:
            print(f"\n[Error] Frame {i} failed: {e}")
            import traceback
            traceback.print_exc()

    cam_writer.close()
    bev_writer.close()
    print("\n✓ Finished! Writers closed safely.")

def get_scene_frames(dataset, max_frames=20):
    print("Scanning dataset for the first available scene...")
    first_token = None
    possible_keys = ['scene_token', 'scene', 'scene_id', 'token']
    for key in possible_keys:
        if len(dataset.data_infos) > 0 and key in dataset.data_infos[0]:
            first_token = dataset.data_infos[0][key]
            print(f"Found scene identifier using key '{key}': {first_token}")
            break
    
    if not first_token:
        print("No scene token found, falling back to sequential frames.")
        total_len = len(dataset.data_infos)
        safe_max = min(total_len, max_frames)
        return list(range(safe_max))

    scene_indices = []
    target_key = None
    for key in possible_keys:
        if key in dataset.data_infos[0]:
            target_key = key
            break
            
    for i, info in enumerate(dataset.data_infos):
        if info.get(target_key) == first_token:
            scene_indices.append(i)
            
    scene_indices.sort(key=lambda i: dataset.data_infos[i].get('timestamp', 0))
    if max_frames and len(scene_indices) > max_frames:
        scene_indices = scene_indices[:max_frames]
        
    print(f"Processing Scene ({first_token}): {len(scene_indices)} frames")
    return scene_indices

def main():
    print("Loading 'orion_stage3_agent.py' ...")
    cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_agent.py')
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    cfg.data.test.test_mode = True
    cfg.model.fp16_infer = False
    if hasattr(cfg, 'fp16'): cfg.fp16 = None
    cfg.model.use_gen_token = True
    cfg.model.use_diff_decoder = False
    cfg = force_disable_flash_attn(cfg)

    print("="*80)
    print("ORION STREAMING DEMO V78 (FIX: FLATTENED MEDIAN + GROUND SHIFT)")
    print("="*80)
    
    print("Building model...")
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    print("Loading checkpoint...")
    load_checkpoint(model, 'ckpts/Orion.pth', map_location='cpu')
    
    model.cuda()
    model.eval()
    model.float()
    
    apply_safe_multimodal_patch(model)
    apply_safe_test_patch(model)
    
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = False
        
    print("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    
    print("Getting frames...")
    indices = get_scene_frames(dataset, max_frames=10)
    
    run_safe_inference_stream(dataset, indices, model)
    
    print("="*80)
    print("DEMO COMPLETE")
    print("Check: stream_orion_cam.gif and stream_orion_bev.gif")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] User interrupted execution.")
    except Exception as e:
        print(f"\n[!] Critical Error: {e}")