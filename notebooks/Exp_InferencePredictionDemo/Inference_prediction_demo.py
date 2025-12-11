# ========================================================================
# ORION INFERENCE DEMO V36 (DIRECT VLM TRAJECTORY + 10 FRAMES)
# ========================================================================
# 0. CRITICAL THREADING FIX (MUST BE FIRST)
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
import imageio
import gc
import sys
import types
import re 
from tqdm import tqdm
from math import factorial
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from mmcv.datasets import build_dataset
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- ORION Placeholder Class ---
class ORION: pass

# ========================================================================
# HELPER: BEZIER INTERPOLATION
# ========================================================================
def bezier_interpolation(controls, n_points=50):
    """
    Interpolates sparse Bezier control points into dense lane lines.
    """
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
# 1. HARD PATCH DEFINITIONS (MODEL LOGIC)
# ========================================================================

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

# --- PATCH 1: Safe Multimodal Preparation ---
def safe_prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes=None
):
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
    print("\n[Hard Patch] Scanning model modules for 'prepare_inputs_labels_for_multimodal'...")
    count = 0
    def recursive_apply(module):
        nonlocal count
        if hasattr(module, 'prepare_inputs_labels_for_multimodal'):
            module.prepare_inputs_labels_for_multimodal = types.MethodType(safe_prepare_inputs_labels_for_multimodal, module)
            count += 1
        for child in module.children():
            recursive_apply(child)
    recursive_apply(model)
    print(f"[Hard Patch] Multimodal preparation patch applied to {count} instances.")

# --- PATCH 2: Safe simple_test_pts (DIRECT VLM MODE) ---
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

            if isinstance(outs, dict):
                 pts_outs.update(outs)
    
    lane_results = None
    vision_embeded_map = None
    if self.with_map_head:
        outs, map_query = self.map_head(img_metas, pos_embed, **data)
        vision_embeded_map = map_query.clone()
        lane_results = self.map_head.get_bboxes(outs, img_metas)
    
    bbox_pts = pts_outs.get('bbox_pts', None)
    metric_dict = pts_outs.get('metric_dict', {})
    generated_text = []
    
    ego_fut_preds = torch.zeros((6, 2), device=device)
    
    if self.with_lm_head and vision_embeded_obj is not None and vision_embeded_map is not None:
        history_input_output_id = []
        vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) 
        input_ids_list = data.get('input_ids', [[]])[0]
        
        for i, input_ids in enumerate(input_ids_list):
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() == 0: 
                     input_ids = input_ids.unsqueeze(0).unsqueeze(0)
                elif input_ids.dim() == 1: 
                     input_ids = input_ids.unsqueeze(0)
            
            # --- DIRECT VLM MODE (SKIPPING MOTION HEAD) ---
            history_input_output_id.append(input_ids)
            context_input_ids = torch.cat(history_input_output_id,dim=-1)
            output_ids = self.lm_head.generate(
                inputs=context_input_ids,
                images=vision_embeded,
                do_sample=False, 
                num_beams=1,
                max_new_tokens=320,
                use_cache=True
            )
            safe_Q = "VLM Label Missing (Inference Mode)"
            vlm_labels_data = img_metas[0].get('vlm_labels')
            if vlm_labels_data is not None and hasattr(vlm_labels_data, 'data') and len(vlm_labels_data.data) > i:
                safe_Q = vlm_labels_data.data[i] 
            generated_text.append(dict(
                Q=safe_Q,
                A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True),
            ))

    # Parse VLM Text to extract Trajectory
    if len(generated_text) > 0:
        traj = generated_text[0]['A'][0]
        # Regex to find [PT, (x,y), ...]
        full_match = re.search(r'\[PT, \(([\+\-]?[\d\.]+), ([\+\-]?[\d\.]+)\)(, \(([\+\-]?[\d\.]+), ([\+\-]?[\d\.]+)\))*\]', traj)
        if full_match:
            coordinates_matches = re.findall(r'\(([\+\-]?[\d\.]+), ([\+\-]?[\d\.]+)\)', full_match.group(0))
            coordinates = [tuple(map(float, coord)) for coord in coordinates_matches]
            coordinates_array = np.array(coordinates)
            ego_fut_preds = torch.tensor(coordinates_array).to(device)
            if len(ego_fut_preds) != 6: 
                ego_fut_preds = torch.zeros((6,2), device=device)

    # Accumulate relative coordinates
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
    is_orion_instance = isinstance(model, ORION)
    is_wrapped_orion = hasattr(model, 'module') and isinstance(model.module, ORION)
    if is_orion_instance:
        model.simple_test_pts = types.MethodType(safe_simple_test_pts, model)
        print("\n[Hard Patch] Successfully replaced ORION.simple_test_pts with safe version.")
    elif is_wrapped_orion:
        model.module.simple_test_pts = types.MethodType(safe_simple_test_pts, model.module)
        print("\n[Hard Patch] Successfully replaced ORION.module.simple_test_pts with safe version.")
    else:
        model.simple_test_pts = types.MethodType(safe_simple_test_pts, model)
        print("\n[Hard Patch] Applied simple_test_pts patch to the top-level model object (assuming ORION structure).")

# --- PATCH 3: Safe simple_test ---
def safe_simple_test(self, img_metas, **data):
    bbox_list, generated_text, lane_results, metric_dict = self.simple_test_pts(img_metas, **data)
    if bbox_list is None:
        bbox_list = [[]] * len(img_metas) 
    final_result = {
        'pts_bbox': {
            'bbox_pts': bbox_list, 
            'lane_results': lane_results,
            'generated_text': generated_text,
            'metric_dict': metric_dict
        }
    }
    return [final_result] 

def apply_safe_test_patch(model):
    is_orion_instance = isinstance(model, ORION)
    is_wrapped_orion = hasattr(model, 'module') and isinstance(model.module, ORION)
    apply_safe_test_pts_patch(model)
    if is_orion_instance:
        model.simple_test = types.MethodType(safe_simple_test, model)
        print("[Hard Patch] Successfully replaced ORION.simple_test with safe output assembly version.")
    elif is_wrapped_orion:
        model.module.simple_test = types.MethodType(safe_simple_test, model.module)
        print("[Hard Patch] Successfully replaced ORION.module.simple_test with safe output assembly version.")
    else:
        model.simple_test = types.MethodType(safe_simple_test, model)
        print("[Hard Patch] Applied simple_test patch to the top-level model object (safe output assembly).")

# ========================================================================
# 2. DATA UTILS & 3. SAFETY PATCHES
# ========================================================================

def aggressive_unwrap(data):
    if hasattr(data, 'data') and not isinstance(data, (torch.Tensor, np.ndarray, str)):
        return aggressive_unwrap(data.data)
    if isinstance(data, list):
        return [aggressive_unwrap(x) for x in data]
    if isinstance(data, tuple):
        return tuple(aggressive_unwrap(x) for x in data)
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
    if isinstance(obj, list) and len(obj) == 1: obj = obj[0]
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
    if name == 'input_ids':
        return obj.cuda(non_blocking=False).long()
    return obj.cuda(non_blocking=False).float()

def extract_field(data_batch, key):
    if key not in data_batch: return None
    return data_batch[key]

def force_disable_flash_attn(cfg):
    print("\n[Config Patch] Disabling Flash Attention...")
    if 'img_backbone' in cfg.model: cfg.model.img_backbone.flash_attn = False
    if 'map_head' in cfg.model and 'transformer' in cfg.model.map_head:
        cfg.model.map_head.transformer.flash_attn = False
    if 'pts_bbox_head' in cfg.model:
        if 'transformer' in cfg.model.pts_bbox_head:
            cfg.model.pts_bbox_head.transformer.flash_attn = False
    return cfg

# ========================================================================
# 4. VISUALIZATION (ENHANCED OPENCV - V36)
# ========================================================================

def render_cam_frame(real_data, trajectory, bbox_results, frame_idx, total):
    img_t = real_data['img']
    if img_t.dim() == 5: img_t = img_t[0, 0] 
    elif img_t.dim() == 4: img_t = img_t[0]
    
    img = img_t.detach().cpu().float().numpy()
    
    # --- FIX 1: De-normalize Image (ImageNet stats) ---
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    img = np.ascontiguousarray(np.transpose(img, (1, 2, 0))) # H,W,C, RGB
    
    l2i = real_data['lidar2img']
    if l2i.dim() == 4: l2i = l2i[0, 0]
    elif l2i.dim() == 3: l2i = l2i[0]
    l2i_np = l2i.detach().cpu().float().numpy()
    
    # --- Draw Bounding Boxes (Projected) ---
    if bbox_results is not None and len(bbox_results) > 0:
        try:
            bboxes_3d = bbox_results[0][0] # tensor (N, 7) or (N, 9)
            if isinstance(bboxes_3d, torch.Tensor):
                 bboxes_3d = bboxes_3d.detach().cpu().numpy()
            
            for box in bboxes_3d:
                x, y, z, w, l, h, rot = box[:7]
                center_hom = np.array([x, y, z, 1.0])
                center_img = np.dot(l2i_np, center_hom)
                if center_img[2] > 0.1: 
                    cx = int(center_img[0] / center_img[2])
                    cy = int(center_img[1] / center_img[2])
                    if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
                        cv2.circle(img, (cx, cy), 4, (0, 165, 255), -1) # Orange Dot
        except:
            pass 

    # --- Draw Trajectory ---
    if trajectory is not None and trajectory.dim() == 2 and trajectory.shape[0] > 0:
        traj_np = trajectory.detach().cpu().float().numpy()
        ones = np.ones((traj_np.shape[0], 1))
        zeros = np.zeros((traj_np.shape[0], 1))
        pts_hom_np = np.concatenate([traj_np, zeros, ones], axis=1) # (6, 4)
        pts_img_np = np.dot(l2i_np, pts_hom_np.T).T # (6, 4)
        
        depth = pts_img_np[:, 3:4]
        depth[np.abs(depth) < 1e-6] = 1e-6
        pts_2d_np = pts_img_np[:, :2] / depth 
        
        h, w, _ = img.shape
        valid_mask = (pts_2d_np[:,0] >= 0) & (pts_2d_np[:,0] < w) & (pts_2d_np[:,1] >= 0) & (pts_2d_np[:,1] < h)
        
        if np.any(valid_mask):
            pts_2d_int = pts_2d_np.astype(np.int32)
            for i in range(len(pts_2d_int) - 1):
                if valid_mask[i] and valid_mask[i+1]:
                    cv2.line(img, tuple(pts_2d_int[i]), tuple(pts_2d_int[i+1]), (255, 0, 0), 3) 
            
            for i in range(len(pts_2d_int)):
                if valid_mask[i]:
                    cv2.circle(img, tuple(pts_2d_int[i]), 6, (255, 0, 0), 1)

    text = f'Frame {frame_idx+1}/{total}'
    cv2.putText(img, text, (int(img.shape[1]*0.05), int(img.shape[0]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img

def render_bev_frame(trajectories, lane_results, bbox_results, current_idx):
    H, W = 600, 600 # Larger canvas
    bev_img = np.ones((H, W, 3), dtype=np.uint8) * 255
    scale = W / 80.0 # Wide view
    cx, cy = W // 2, H // 2
    def world2pix(x, y):
        u = int(cx + x * scale)
        v = int(cy - y * scale) 
        return (u, v)

    # Draw Grid
    cv2.line(bev_img, (0, cy), (W, cy), (200, 200, 200), 1)
    cv2.line(bev_img, (cx, 0), (cx, H), (200, 200, 200), 1)
    
    # --- FIX 2: COORDINATE SYSTEM ALIGNMENT ---
    if lane_results is not None and len(lane_results) > 0:
        try:
            detection = lane_results[0]
            if 'map_pts_3d' in detection:
                controls = detection['map_pts_3d']
                scores = detection['map_scores_3d']
                if isinstance(controls, torch.Tensor): controls = controls.detach().cpu().numpy()
                if isinstance(scores, torch.Tensor): scores = scores.detach().cpu().numpy()
                
                dense_lanes = bezier_interpolation(controls, n_points=50) # (N, 50, 3)
                
                for i, lane_pts_3d in enumerate(dense_lanes):
                    if scores[i] > 0.3: 
                        x_model = lane_pts_3d[:, 0]
                        y_model = lane_pts_3d[:, 1]
                        pts_pix = np.array([world2pix(-y, x) for x, y in zip(x_model, y_model)], dtype=np.int32)
                        pts_pix = pts_pix.reshape((-1, 1, 2))
                        cv2.polylines(bev_img, [pts_pix], False, (180, 180, 180), 1)
        except Exception:
            pass

    if bbox_results is not None and len(bbox_results) > 0:
        try:
            bboxes_3d = bbox_results[0][0] # (N, 7)
            if isinstance(bboxes_3d, torch.Tensor):
                 bboxes_3d = bboxes_3d.detach().cpu().numpy()
            
            for box in bboxes_3d:
                x, y, z, l, w, h, rot = box[:7]
                uc, vc = world2pix(-y, x) 
                w_pix = max(2, int(w * scale))
                l_pix = max(2, int(l * scale))
                top_left = (uc - w_pix//2, vc - l_pix//2)
                bottom_right = (uc + w_pix//2, vc + l_pix//2)
                cv2.rectangle(bev_img, top_left, bottom_right, (0, 165, 255), 1) 
        except:
            pass

    w_pix = int(2 * scale)
    h_pix = int(3 * scale)
    top_left = (cx - w_pix//2, cy - h_pix//2)
    bottom_right = (cx + w_pix//2, cy + h_pix//2)
    cv2.rectangle(bev_img, top_left, bottom_right, (255, 0, 0), -1) 

    start_history = max(0, current_idx - 20)
    for i in range(start_history, current_idx):
        if trajectories[i].dim() == 2 and trajectories[i].shape[0] > 0:
            traj_np = trajectories[i].detach().cpu().float().numpy()
            pts_pix = [world2pix(-p[1], p[0]) for p in traj_np]
            for j in range(len(pts_pix)-1):
                cv2.line(bev_img, pts_pix[j], pts_pix[j+1], (0, 255, 0), 1) 

    c = trajectories[current_idx]
    if c.dim() == 2 and c.shape[0] > 0:
        c_np = c.detach().cpu().float().numpy()
        pts_pix = [world2pix(-p[1], p[0]) for p in c_np]
        for j in range(len(pts_pix)-1):
            cv2.line(bev_img, pts_pix[j], pts_pix[j+1], (0, 0, 255), 2) 

    return bev_img

def run_safe_inference_stream(dataset, indices, model):  
    print(f"\nInitializing Direct-to-Disk GIF writers...")  
    cam_writer = imageio.get_writer('stream_orion_cam.gif', mode='I', fps=2, loop=0)  
    bev_writer = imageio.get_writer('stream_orion_bev.gif', mode='I', fps=2, loop=0)  
    traj_history = []  
      
    print(f"Starting Stream Processing ({len(indices)} frames)...")  
      
    if torch.cuda.is_available():  
        print("[System] Synchronizing GPU...")  
        torch.cuda.synchronize()  
      
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
                if val is not None:  
                    real_data[k] = process_to_gpu_batch(val, k)
              
            if 'input_ids' not in real_data or real_data['input_ids'] is None:  
                dummy_token = 1  
                real_data['input_ids'] = torch.tensor([[dummy_token]], dtype=torch.long).cuda()  
            else:  
                if real_data['input_ids'].dim() == 1:  
                    real_data['input_ids'] = real_data['input_ids'].unsqueeze(0)  
                real_data['input_ids'] = real_data['input_ids'].cuda().long()
              
            img_metas_raw = extract_field(example, 'img_metas')  
            img_metas = aggressive_unwrap(img_metas_raw)  
            if isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], list):  
                img_metas = img_metas[0]  
            if isinstance(img_metas, dict):  
                img_metas = [img_metas]  
              
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
                     
                    if 'bbox_pts' in pts_data:
                        bbox_results = pts_data['bbox_pts']

                    if 'lane_results' in pts_data:
                        lane_results = pts_data['lane_results']

            traj_history.append(pred_traj)  
            
            cam_frame = render_cam_frame(real_data, pred_traj, bbox_results, i, len(indices))  
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

# --- V36: ROBUST SCENE LOADING (10 FRAMES DEFAULT) ---
def get_scene_frames(dataset, max_frames=10):
    print("Scanning dataset for the first available scene...")
    
    # Debug: Print available keys in the first data info
    if len(dataset.data_infos) > 0:
        print("[Debug] Available keys in data_infos[0]:")
        for key in dataset.data_infos[0].keys():
            print(f"  - {key}")

    # Try different possible keys for scene identification
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
    print("ORION STREAMING DEMO (HARD PATCH v36 - DIRECT VLM + 10 FRAMES)")
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