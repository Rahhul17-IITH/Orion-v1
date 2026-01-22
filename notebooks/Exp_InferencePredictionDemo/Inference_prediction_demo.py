# ========================================================================
# ORION INFERENCE DEMO V99 (FIX: PURE OPENCV RENDERING)
# ========================================================================
# 1. FIX: Removed ALL Matplotlib dependencies (Solves 'object __array__' error).
# 2. FIX: Implemented direct CV2 drawing for Camera and BEV trajectories.
# 3. RETAINED: Pipeline timing and Model stability patches.
# ========================================================================

import cv2
cv2.setNumThreads(0)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["cv_num_threads"] = "1"

# ========================================================================
# IMPORTS
# ========================================================================
import torch
import numpy as np
import io
import inspect
import sys
# REMOVED: All matplotlib imports to prevent backend crashes

# CRITICAL PATCH: Restore np.int
if not hasattr(np, 'int'):
    np.int = int

import imageio
import gc
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

warnings.filterwarnings("ignore")

# --- Constants ---
LIDAR_HEIGHT_CORRECTION = 1.85 

class ORION:
    pass

# ========================================================================
# 1. HELPER STRUCTURES & WRAPPERS
# ========================================================================

# REMOVED: get_configured_fig
# REMOVED: fig_to_numpy
# REMOVED: CustomNuscenesBox (Replaced with direct CV2 logic)

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
            return torch.zeros((self.tensor.shape[0], 8, 3)) 

# ========================================================================
# 2. VIZ HELPERS & PROJECTION
# ========================================================================
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
    if hasattr(bboxes3d, 'tensor'): corners_3d = bboxes3d.corners.detach().cpu().numpy()
    else: corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor): lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=0.1, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
    return plot_rect3d_on_img(raw_img, num_bbox, imgfov_pts_2d, color, thickness)

def show_multi_modality_result(img, gt_bboxes, pred_bboxes, proj_mat, out_dir, filename, show=True, pred_bbox_color=(0, 165, 255)):
    if pred_bboxes is not None:
        img = draw_lidar_bbox3d_on_img(pred_bboxes, img, proj_mat, None, color=pred_bbox_color, thickness=2)
    return img

def bezier_interpolation(controls, n_points=50):
    if len(controls) == 0: return np.zeros((0, n_points, 3))
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
# 3. VISUALIZATION ENGINE (PURE CV2)
# ========================================================================

def render_cam_frame(real_data, trajectory, bbox_results, lane_results, frame_idx, total):
    img_t = real_data['img']
    if img_t.dim() == 5: img_t = img_t[0, 0]
    elif img_t.dim() == 4: img_t = img_t[0]
    img = img_t.detach().cpu().float().numpy()
    
    mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
    std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
    
    img = std * img + mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(np.transpose(img, (1, 2, 0)))
    
    l2i = real_data['lidar2img']
    if l2i.dim() == 4: l2i = l2i[0, 0]
    elif l2i.dim() == 3: l2i = l2i[0]
    l2i_np = l2i.detach().cpu().float().numpy()

    # 1. Bounding Boxes
    if bbox_results is not None and len(bbox_results) > 0:
        try:
            bbox_obj = bbox_results[0][0]
            if hasattr(bbox_obj, 'tensor'): bboxes_3d_tensor = bbox_obj.tensor
            elif hasattr(bbox_obj, 'data'): bboxes_3d_tensor = bbox_obj.data
            else: bboxes_3d_tensor = torch.tensor(bbox_obj)
            
            scores = bbox_results[0][1]
            if isinstance(scores, torch.Tensor): scores = scores.detach().cpu().numpy()
            mask = scores > 0.1 
            
            if mask.any():
                valid_boxes = bboxes_3d_tensor[mask].clone().detach().cpu()
                if valid_boxes.dim() == 1: valid_boxes = valid_boxes.unsqueeze(0)
                lidar_boxes = LiDARInstance3DBoxes(valid_boxes, box_dim=valid_boxes.shape[-1], origin=(0.5, 0.5, 0))
                img = show_multi_modality_result(
                    img=img, gt_bboxes=None, pred_bboxes=lidar_boxes, 
                    proj_mat=l2i_np, out_dir=None, filename=None, show=False
                )
        except Exception as e: pass

    # 2. Trajectories (PURE CV2)
    if trajectory is not None and trajectory.numel() > 0:
        if trajectory.dim() == 3: traj_to_show = trajectory[0] # Mode 0
        else: traj_to_show = trajectory

        if torch.max(torch.abs(traj_to_show)) > 0.001:
            traj_np = traj_to_show.detach().cpu().float().numpy()
            z_vals = np.full((traj_np.shape[0], 1), -LIDAR_HEIGHT_CORRECTION)
            traj_3d = np.hstack((traj_np, z_vals))
            
            pts_2d_depth = points_cam2img(traj_3d, l2i_np, with_depth=True)
            mask = pts_2d_depth[:, 2] > 0.1
            valid_pts = pts_2d_depth[mask][:, :2]
            
            # --- CV2 RENDERING ---
            if len(valid_pts) > 1:
                # Convert to int32 for cv2
                pts_int = valid_pts.astype(np.int32).reshape((-1, 1, 2))
                # Draw Trajectory Line
                cv2.polylines(img, [pts_int], isClosed=False, color=(0, 0, 255), thickness=3)
                # Draw End Point
                end_pt = tuple(pts_int[-1][0])
                cv2.circle(img, end_pt, 6, (0, 0, 255), -1)
            # ---------------------

    text = f'Frame {frame_idx+1}/{total}'
    cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def render_bev_frame(trajectories, lane_results, bbox_results, current_idx):
    H, W = 800, 800
    bev_img = np.ones((H, W, 3), dtype=np.uint8) * 30
    scale = W / 100.0
    cx, cy = W // 2, H // 2 + 200

    def world2pix(x, y):
        u = int(cx + x * scale)
        v = int(cy - y * scale)
        return (u, v)

    # Grid
    for i in range(-20, 120, 20):
        cv2.line(bev_img, world2pix(-50, i), world2pix(50, i), (60, 60, 60), 1)
    for i in range(-50, 50, 10):
        cv2.line(bev_img, world2pix(i, 0), world2pix(i, 100), (60, 60, 60), 1)

    # Calibration Markers
    origin_pix = world2pix(0, 0)
    cv2.drawMarker(bev_img, origin_pix, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    
    # Lanes
    if lane_results is not None and len(lane_results) > 0:
        try:
            detection = lane_results[0]
            if 'map_pts_3d' in detection:
                controls = detection['map_pts_3d'].detach().cpu().numpy()
                scores = detection['map_scores_3d'].detach().cpu().numpy()
                dense_lanes = bezier_interpolation(controls, n_points=50)
                for i, lane_pts_3d in enumerate(dense_lanes):
                    if scores[i] > 0.45:
                        pts_pix = np.array([world2pix(pt[0], pt[1]) for pt in lane_pts_3d], dtype=np.int32)
                        cv2.polylines(bev_img, [pts_pix.reshape((-1, 1, 2))], False, (100, 255, 100), 2)
        except: pass

    # Boxes
    if bbox_results is not None and len(bbox_results) > 0:
        try:
            bbox_obj = bbox_results[0][0]
            if hasattr(bbox_obj, 'tensor'): bboxes_3d = bbox_obj.tensor
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
        except Exception as e: pass

    # 3. Trajectory (PURE CV2)
    c = trajectories[current_idx]
    if c is not None and c.numel() > 0:
        traj_cpu = c.detach().cpu().numpy()
        num_modes = traj_cpu.shape[0]
        
        # Colors for different modes (BGR)
        colors = [
            (255, 0, 0),   # Blue
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (0, 255, 255), # Yellow
            (255, 0, 255), # Magenta
            (255, 255, 0)  # Cyan
        ]

        for m in range(num_modes):
            traj_mode = traj_cpu[m]
            
            # Unroll relative coords if necessary
            if np.abs(traj_mode).max() < 10.0:
                 traj_mode = traj_mode.cumsum(axis=0)
            
            # --- FIX: DIRECT COORDINATE TRANSFORMATION ---
            pts_pix = []
            for pt in traj_mode:
                pts_pix.append(world2pix(pt[0], pt[1]))
            
            pts_pix = np.array(pts_pix, dtype=np.int32).reshape((-1, 1, 2))
            
            # Use specific color for each mode, cycle if more modes than colors
            color = colors[m % len(colors)]
            thickness = 2 if m == 0 else 1 # Highlight first mode
            
            cv2.polylines(bev_img, [pts_pix], isClosed=False, color=color, thickness=thickness)
            
            # Draw end point
            end_pt = tuple(pts_pix[-1][0])
            cv2.circle(bev_img, end_pt, 4, color, -1)
            # ---------------------------------------------

    # Ego Car
    ego_u, ego_v = world2pix(0, 0)
    pts_car = np.array([
        [ego_u, ego_v - 15], [ego_u + 10, ego_v + 10],  
        [ego_u, ego_v + 5], [ego_u - 10, ego_v + 10]    
    ], dtype=np.int32)
    cv2.fillPoly(bev_img, [pts_car], (255, 0, 0))
    
    return bev_img

# ========================================================================
# 4. MODEL PATCHES (CRITICAL UPDATES)
# ========================================================================
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

# --- FIX: ROBUST TENSOR COMPARISON & NONE CHECK ---
def inference_ego_patched(self, inputs=None, images=None, image_sizes=None, return_ego_feature=False, **kwargs):
    """Corrected version of mmcv/utils/llava_llama.py:inference_ego"""
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        (inputs, position_ids, attention_mask, _, inputs_embeds, _, new_input_ids) = self.prepare_inputs_labels_for_multimodal(
            inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
        )
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)

    output_attentions = self.config.output_attentions
    output_hidden_states = self.config.output_hidden_states
    return_dict = self.config.use_return_dict

    outputs = self.model(
        input_ids=inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    if return_ego_feature:
        # --- FALLBACK IF new_input_ids IS NONE ---
        if new_input_ids is None:
            new_input_ids = inputs
        # -----------------------------------------------

        if not isinstance(self.config.waypoint_token_idx, list):
            # --- USE as_tensor (Safe for Lists) ---
            if not isinstance(new_input_ids, torch.Tensor):
                new_input_ids = torch.as_tensor(new_input_ids, device=hidden_states.device)
            # --------------------------------------------
            
            loc_positions = (new_input_ids == self.config.waypoint_token_idx)
            
            # --- REMOVE device= KEYWORD (SYNTAX ERROR) ---
            loc_positions = loc_positions.to(hidden_states.device)
            # ----------------------------------------------------
            
            selected_hidden_states = hidden_states[loc_positions]
        else:
            loc_positions_list = []
            for new_id in new_input_ids:
                # --- USE as_tensor (Safe for Lists) ---
                if not isinstance(new_id, torch.Tensor):
                    new_id = torch.as_tensor(new_id, device=hidden_states.device)
                # --------------------------------------------

                loc_positions = torch.zeros_like(new_id).to(torch.bool)
                for token_id in self.config.waypoint_token_idx:
                    if token_id in new_id:
                        loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                loc_positions_list.append(loc_positions)
            loc_positions = torch.stack(loc_positions_list, dim=0)
            
            # --- REMOVE device= KEYWORD ---
            loc_positions = loc_positions.to(hidden_states.device)
            # -----------------------------------
            
            selected_hidden_states = hidden_states[loc_positions]
        return selected_hidden_states

    return outputs

def apply_explicit_patch():
    """Apply the explicit function above to the library class globally"""
    # PATCHING METHOD: MODULE LEVEL (Most Reliable)
    # This affects the class definition in sys.modules, so any subsequent 
    # instantiation (like build_model) picks up the patched method.
    try:
        import mmcv.utils.llava_llama as lib_module
        lib_module.LlavaLlamaForCausalLM.inference_ego = inference_ego_patched
        print("[Patch] SUCCESS: Explicitly replaced LlavaLlamaForCausalLM.inference_ego at MODULE level")
    except ImportError:
        print("[Patch] Warning: Library mmcv.utils.llava_llama not found, trying local scope.")
    except Exception as e:
        print(f"[Patch] FAILURE: {e}")

def apply_safe_multimodal_patch(model):
    # DISABLED CUSTOM PATCH (Using Built-in)
    print("\n[Hard Patch] Skipped custom multimodal patch (Using built-in implementation)...")
    pass

def safe_simple_test_pts(self, img_metas, **data):
    # FIXED V87: Full VAE Decoding + Multi-Mode Output
    device = next(self.parameters()).device
    
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
        
        # Initialize as 3D tensor to hold modes: [6, 6, 2] -> Modes, Time, 2
        ego_fut_preds = torch.zeros((6, 6, 2), device=device)

        if self.with_lm_head and vision_embeded_obj is not None and vision_embeded_map is not None:
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
            input_ids_list = data.get('input_ids', [[]])[0]
            
            for i, input_ids in enumerate(input_ids_list):
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() == 0: input_ids = input_ids.unsqueeze(0).unsqueeze(0)
                    elif input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
                
                special_token_inputs = False
                if hasattr(self.lm_head.config, 'waypoint_token_idx'):
                    special_token_inputs = self.lm_head.config.waypoint_token_idx in input_ids

                if self.use_gen_token and special_token_inputs:
                    ego_feature = self.lm_head.inference_ego(
                        inputs=input_ids,
                        images=vision_embeded,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True,
                        return_ego_feature=True
                    )
                    
                    if not self.use_diff_decoder and not self.use_mlp_decoder:
                        current_states = ego_feature.unsqueeze(1)
                        B = current_states.shape[0]
                        
                        distribution_comp = {}
                        noise = None
                        sample = None
                        if hasattr(self, 'PROBABILISTIC') and self.PROBABILISTIC:
                            sample, output_distribution = self.distribution_forward(
                                current_states, None, noise
                            )
                            distribution_comp = {**distribution_comp, **output_distribution}
                        
                        hidden_states = ego_feature.unsqueeze(1)
                        if sample is None: 
                            sample = torch.randn((B, self.latent_dim), device=device)
                            
                        states_hs, future_states_hs = self.future_states_predict(
                            B, sample, hidden_states, current_states
                        )
                        
                        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                        ego_fut_trajs_list = []
                        fut_ts = getattr(self, 'fut_ts', 6)
                        
                        for t in range(fut_ts):
                            outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[t]).reshape(B, self.ego_fut_mode, 2)
                            ego_fut_trajs_list.append(outputs_ego_trajs)
                        
                        # [B, T, Modes, 2] -> [1, 6, 6, 2]
                        ego_fut_preds_modes = torch.stack(ego_fut_trajs_list, dim=2) 
                        
                        # Return all modes [Modes, Time, 2]
                        ego_fut_preds = ego_fut_preds_modes[0].permute(1, 0, 2).to('cpu') 
                    else:
                        ego_fut_preds = torch.zeros((6, 6, 2), device=device)

        # Accumulate if relative
        if torch.max(torch.abs(ego_fut_preds)) < 10.0:
            ego_fut_preds = ego_fut_preds.cumsum(dim=1) # Cumsum along Time dimension

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
# 5. RUN LOOP
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
    
    # Get Waypoint Token Index for Manual Injection
    waypoint_token = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head.config, 'waypoint_token_idx'):
        waypoint_token = model.lm_head.config.waypoint_token_idx
        if isinstance(waypoint_token, list): waypoint_token = waypoint_token[0]
        elif isinstance(waypoint_token, torch.Tensor): waypoint_token = waypoint_token.item()
        print(f"Detected Waypoint Token Index: {waypoint_token}")

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
            
            # --- AGGRESSIVE FIX: FORCE Token Injection ---
            if 'input_ids' in real_data and waypoint_token is not None:
                inp = real_data['input_ids']
                if not (inp == waypoint_token).any():
                    new_token = torch.tensor([[waypoint_token]], dtype=inp.dtype, device=inp.device)
                    real_data['input_ids'] = torch.cat([inp, new_token], dim=1)
            elif 'input_ids' not in real_data and waypoint_token is not None:
                prompt_text = "Please provide the planning trajectory for the ego car without reasons."
                tokenizer = getattr(model, 'tokenizer', getattr(getattr(model, 'module', None), 'tokenizer', None))
                if tokenizer:
                    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                    tokens.append(waypoint_token)
                    real_data['input_ids'] = torch.tensor([tokens], dtype=torch.long).cuda()
            
            if 'input_ids' in real_data:
                real_data['input_ids'] = real_data['input_ids'].cuda().long()

            img_metas_raw = extract_field(example, 'img_metas')
            img_metas = aggressive_unwrap(img_metas_raw)
            if isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], list):
                img_metas = img_metas[0]
            if isinstance(img_metas, dict): img_metas = [img_metas]

            with torch.no_grad():
                result = model.simple_test(img_metas, **real_data)
                
            # pred_traj will now be [Modes, Time, 2]
            pred_traj = torch.zeros((6, 6, 2)) 
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
            
            # V99: Using pure OpenCV renderers
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
    # --- CRITICAL: APPLY PATCH BEFORE MODEL BUILD ---
    apply_explicit_patch()
    
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
    
    # 4. Verification Check
    if hasattr(cfg, 'img_norm_cfg'):
        print(f"Config Norm: {cfg.img_norm_cfg}")

    print("="*80)
    print("ORION STREAMING DEMO V99 (FIXED: PURE OPENCV)")
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
    # --- FIX: PIPELINE SWITCH BEFORE DATASET BUILD ---
    if hasattr(cfg, 'inference_only_pipeline'):
        print(f"Switching pipeline to: inference_only_pipeline (DeepWiki Rec)")
        cfg.data.test.pipeline = cfg.inference_only_pipeline
        
    dataset = build_dataset(cfg.data.test)
    # -----------------------------------------------------
    
    print("Getting frames...")
    indices = get_scene_frames(dataset, max_frames=20)
    
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