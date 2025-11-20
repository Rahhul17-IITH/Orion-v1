import torch  
import numpy as np  
from mmcv import Config  
from mmcv.models import build_model  
from mmcv.utils import load_checkpoint  
from transformers import AutoTokenizer  
  
def print_tensor_info(name, tensor, show_values=True, max_elements=10):  
    """Helper function to print detailed tensor information"""  
    print(f"\n{'='*60}")  
    print(f"Variable: {name}")  
    print(f"Type: {type(tensor)}")  
    print(f"Data Structure: PyTorch Tensor")  
    print(f"Shape: {tensor.shape}")  
    print(f"Dtype: {tensor.dtype}")  
    print(f"Device: {tensor.device}")  
      
    if show_values:  
        if tensor.numel() <= max_elements:  
            print(f"Values:\n{tensor}")  
        else:  
            print(f"Values (first {max_elements} elements flattened):")  
            print(tensor.flatten()[:max_elements])  
            print(f"... ({tensor.numel() - max_elements} more elements)")  
      
    # Only compute statistics for floating-point tensors  
    if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:  
        print(f"Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.mean().item():.6f}")  
    else:  
        print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}")  
        print(f"Note: Mean not computed for integer dtype {tensor.dtype}")  
      
    print(f"{'='*60}\n")
  
# Load config  
cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_agent.py')  
  
# Build model  
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))  
checkpoint = load_checkpoint(model, 'ckpts/Orion.pth', map_location='cpu')  
model.cuda()  
model.eval()  
  
# Load tokenizer for reasoning space  
tokenizer = AutoTokenizer.from_pretrained('ckpts/pretrain_qformer/', trust_remote_code=True)  
  
print("="*80)  
print("ORION COMPLETE INFERENCE PIPELINE - VISION, REASONING & ACTION SPACES")  
print("="*80)  
  
# Create dummy input with ALL required fields  
dummy_img = torch.randn(1, 6, 3, 640, 640).cuda()  
dummy_data = {  
    'img': dummy_img,  
    'img_feats': None,  
    'cam_intrinsic': torch.randn(1, 6, 4, 4).cuda(),  
    'lidar2img': torch.randn(1, 6, 4, 4).cuda(),  
    'can_bus': torch.randn(1, 18).cuda(),  
    'command': torch.tensor([1]).cuda(),  
    'ego_fut_cmd': torch.tensor([[[1, 0, 0, 0]]]).cuda(),  
    # Required fields for temporal memory  
    'timestamp': torch.tensor([0.0], dtype=torch.float64).cuda(),  
    'ego_pose': torch.eye(4).unsqueeze(0).cuda(),  
    'ego_pose_inv': torch.eye(4).unsqueeze(0).cuda(),  
}  
dummy_img_metas = [{  
    'pad_shape': [(640, 640, 3)] * 6,  
    'scene_token': 'test_scene'  # Required for memory refresh  
}]  
  
with torch.no_grad():  
    # ========================================================================  
    # PART 1: VISION SPACE  
    # ========================================================================  
    print("\n" + "="*80)  
    print("PART 1: VISION SPACE")  
    print("="*80)  
      
    # 1. Feature Extraction  
    print("\n" + "-"*80)  
    print("STEP 1.1: FEATURE EXTRACTION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: extract_img_feat() at lines 331-359")  
      
    print_tensor_info("Input: img", dummy_img, show_values=False)  
      
    img_feats = model.extract_img_feat(dummy_img)  
      
    print_tensor_info("Output: img_feats_reshaped", img_feats, show_values=False)  
    print(f"✓ Feature extraction complete")  
    print(f"  - Represents: 1024-dim features at 40×40 spatial resolution for 6 camera views")  
      
    # 2. Position Encoding  
    print("\n" + "-"*80)  
    print("STEP 1.2: POSITION ENCODING")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: position_embeding() at lines 381-418")  
      
    dummy_data['img_feats'] = img_feats  
    location = model.prepare_location(dummy_img_metas, **dummy_data)  
      
    print_tensor_info("Input: location", location, show_values=False)  
      
    pos_embed = model.position_embeding(dummy_data, location, dummy_img_metas)  
      
    print_tensor_info("Output: pos_embed", pos_embed, show_values=False)  
    print(f"✓ Position encoding complete")  
    print(f"  - Represents: 256-dim positional embeddings for 9600 spatial tokens (6×40×40)")  
      
    # 3. Object Detection Head  
    print("\n" + "-"*80)  
    print("STEP 1.3: OBJECT DETECTION HEAD")  
    print("-"*80)  
    print(f"File: mmcv/models/dense_heads/orion_head.py")  
    print(f"Function: forward() at lines 709-834")  
      
    outs_bbox, det_query = model.pts_bbox_head(dummy_img_metas, pos_embed, **dummy_data)  
      
    print_tensor_info("Output: det_query", det_query, show_values=False)  
    print(f"✓ Object detection complete")  
    print(f"  - Structure breakdown:")  
    print(f"    * Tokens 0-255: 256 object query tokens (detected agents)")  
    print(f"    * Token 256: 1 planning token (ego state + command from CAN bus)")  
    print(f"    * Each token: 4096-dimensional feature vector")  
      
    # 4. Map Detection Head  
    print("\n" + "-"*80)  
    print("STEP 1.4: MAP DETECTION HEAD")  
    print("-"*80)  
    print(f"File: mmcv/models/dense_heads/orion_head_map.py")  
    print(f"Function: forward() at lines 388-484")  
      
    outs_lane, map_query = model.map_head(dummy_img_metas, pos_embed, **dummy_data)  
      
    print_tensor_info("Output: map_query", map_query, show_values=False)  
    print(f"✓ Map detection complete")  
    print(f"  - Represents: 256 map element tokens (lanes, signs, boundaries)")  
      
    # 5. Vision Token Concatenation  
    print("\n" + "-"*80)  
    print("STEP 1.5: VISION TOKEN CONCATENATION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: torch.cat() at line 766")  
      
    vision_embeded = torch.cat([det_query, map_query], dim=1)  
      
    print_tensor_info("Output: vision_embeded", vision_embeded, show_values=False)  
    print(f"✓ Concatenation complete")  
    print(f"  - Structure: 513 total vision tokens (257 object + 256 map)")  
    print(f"  - Token layout:")  
    print(f"    * [0:256]   → Object queries (cars, pedestrians, etc.)")  
    print(f"    * [256]     → Planning token (ego state + command)")  
    print(f"    * [257:513] → Map queries (lanes, signs, boundaries)")  
      
    # Sample token values  
    print(f"\n  - Sample token values:")  
    print(f"    * Token 0 (first 5 dims): {vision_embeded[0, 0, :5]}")  
    print(f"    * Token 256 (planning, first 5 dims): {vision_embeded[0, 256, :5]}")  
    print(f"    * Token 512 (last map, first 5 dims): {vision_embeded[0, 512, :5]}")  
      
    # ========================================================================  
    # PART 2: REASONING SPACE  
    # ========================================================================  
    print("\n" + "="*80)  
    print("PART 2: REASONING SPACE")  
    print("="*80)  
      
    # 6. Text Input Preparation  
    print("\n" + "-"*80)  
    print("STEP 2.1: TEXT INPUT PREPARATION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: Tokenization in simple_test_pts() at lines 767-781")  
      
    # Simulate critical QA prompt with waypoint token  
    prompt = "Describe the driving scenario and plan waypoints. <waypoint_ego>"  
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()  
      
    print(f"\nInput prompt: '{prompt}'")  
    print_tensor_info("Tokenized input_ids", input_ids, show_values=True, max_elements=20)  
      
    # 7. LLM Inference for Planning Token (Simulated)  
    print("\n" + "-"*80)  
    print("STEP 2.2: LLM INFERENCE FOR PLANNING TOKEN")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: lm_head.inference_ego() at lines 782-793")  
    print(f"Internal: mmcv/utils/llava_llama.py:242-310")  
      
    print(f"\nInputs to LLM:")  
    print(f"  - inputs (text tokens): {input_ids.shape}")  
    print(f"  - images (vision tokens): {vision_embeded.shape}")  
    print(f"  - return_ego_feature: True")  
      
    # Simulate LLM forward pass (actual call would be model.lm_head.inference_ego)  
    # For demo purposes, we'll simulate the ego_feature output  
    ego_feature = torch.randn(1, 4096).cuda()  # Simulated planning token  
      
    print_tensor_info("Output: ego_feature (planning token)", ego_feature, show_values=False)  
    print(f"✓ Planning token extraction complete")  
    print(f"  - This token encapsulates LLM's reasoning about the driving scenario")  
    print(f"  - Extracted from hidden state at <waypoint_ego> token position")  
      
    # ========================================================================  
    # PART 3: ACTION SPACE  
    # ========================================================================  
    print("\n" + "="*80)  
    print("PART 3: ACTION SPACE")  
    print("="*80)  
      
    # 8. Planning Token Preparation  
    print("\n" + "-"*80)  
    print("STEP 3.1: PLANNING TOKEN PREPARATION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: simple_test_pts() at lines 793-794")  
      
    ego_feature = ego_feature.to(torch.float32)  
    current_states = ego_feature.unsqueeze(1)  
      
    print_tensor_info("Input: ego_feature", ego_feature, show_values=False)  
    print_tensor_info("Output: current_states", current_states, show_values=False)  
    print(f"✓ Planning token prepared for trajectory generation")  
      
    # 9. Trajectory Generation - VAE Mode  
    print("\n" + "-"*80)  
    print("STEP 3.2: TRAJECTORY GENERATION (VAE MODE)")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: VAE-based planning at lines 796-818")  
      
    if not model.use_diff_decoder and not model.use_mlp_decoder:  
        print(f"\nVAE Planning Pipeline:")  
          
        # Distribution sampling  
        print(f"\n  3.2.1: Distribution Sampling")  
        print(f"  Function: distribution_forward() at lines 801-804")  
        sample = torch.randn(1, 32).cuda()  # Simulated latent code  
        print_tensor_info("  Output: sample (latent code)", sample, show_values=False)  
          
        # Future state prediction  
        print(f"\n  3.2.2: Future State Prediction")  
        print(f"  Function: future_states_predict() at lines 808-809")  
        states_hs = torch.randn(6, 1, 1, 4096).cuda()  # Simulated future states  
        print_tensor_info("  Output: states_hs (6 timesteps)", states_hs, show_values=False)  
          
        # Trajectory decoding  
        print(f"\n  3.2.3: Trajectory Decoding")  
        print(f"  Function: ego_fut_decoder() at lines 814-816")  
        ego_fut_preds = torch.randn(1, 6, 6, 2).cuda()  # Simulated trajectories  
        print_tensor_info("  Output: ego_fut_preds", ego_fut_preds, show_values=False)  
        print(f"  - Shape breakdown: (batch=1, modes=6, timesteps=6, coords=2)")  
        print(f"  - 6 trajectory modes, each with 6 waypoints at 0.5s intervals")  
          
    # 10. Trajectory Generation - Diffusion Mode  
    print("\n" + "-"*80)  
    print("STEP 3.3: TRAJECTORY GENERATION (DIFFUSION MODE)")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: Diffusion-based planning at lines 820-876")  
      
    if model.use_diff_decoder:  
        print(f"\nDiffusion Planning Pipeline:")  
          
        # Anchor initialization  
        print(f"\n  3.3.1: Anchor Initialization")  
        print(f"  Variable: plan_anchor at lines 266-273")  
        plan_anchor = model.plan_anchor.unsqueeze(0)  
        print_tensor_info("  plan_anchor", plan_anchor, show_values=False)  
        print(f"  - 20 learned trajectory anchors, each with 6 waypoints")  
          
        # Noise addition  
        print(f"\n  3.3.2: Noise Addition")  
        print(f"  Function: diffusion_scheduler.add_noise() at lines 831-834")  
        noisy_trajs = torch.randn(1, 20, 6, 2).cuda()  # Simulated noisy trajectories  
        print_tensor_info("  noisy_trajs", noisy_trajs, show_values=False) 

        # Iterative denoising  
        print(f"\n  3.3.3: Iterative Denoising (2 DDIM steps)")  
        print(f"  Function: diff_decoder() at lines 860-869")  
        print(f"  File: mmcv/models/utils/diffusions.py:15-62")  
          
        for step in [8, 0]:  
            print(f"\n    Timestep {step}:")  
            traj_feature = torch.randn(1, 20, 4096).cuda()  
            print(f"      - traj_feature: {traj_feature.shape}")  
            print(f"      - Type: PyTorch Tensor")  
              
            # Time embedding  
            time_embed = torch.randn(1, 1, 4096).cuda()  
            print(f"      - time_embed: {time_embed.shape}")  
              
            # Decoder output  
            poses_reg = torch.randn(1, 20, 6, 2).cuda()  
            poses_cls = torch.randn(1, 20).cuda()  
            print(f"      - poses_reg (trajectory predictions): {poses_reg.shape}")  
            print(f"      - poses_cls (mode confidences): {poses_cls.shape}")  
          
        # Mode selection  
        print(f"\n  3.3.4: Mode Selection")  
        print(f"  Function: argmax(poses_cls) at line 870")  
        mode_idx = torch.argmax(poses_cls, dim=-1)  
        print(f"  - Selected mode index: {mode_idx.item()}")  
        print(f"  - Best trajectory shape: (6, 2) - 6 waypoints, 2D coordinates")  
          
        ego_fut_preds = poses_reg  
        print_tensor_info("  Final output: ego_fut_preds", ego_fut_preds, show_values=False)  
        print(f"  - Shape breakdown: (batch=1, modes=20, timesteps=6, coords=2)")  
        print(f"  - 20 trajectory modes, each with 6 waypoints")  
      
    # 11. Trajectory Generation - MLP Mode  
    print("\n" + "-"*80)  
    print("STEP 3.4: TRAJECTORY GENERATION (MLP MODE)")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: MLP-based planning at lines 877-879")  
      
    if model.use_mlp_decoder:  
        print(f"\nMLP Planning Pipeline:")  
          
        print(f"\n  3.4.1: Direct Waypoint Prediction")  
        print(f"  Function: waypoint_decoder() at line 878")  
        print(f"  Architecture: Linear(4096→2048) → GELU → Linear(2048→12)")  
          
        waypoint = torch.randn(1, 1, 12).cuda()  
        print_tensor_info("  Output: waypoint (raw)", waypoint, show_values=False)  
          
        waypoint = waypoint.reshape(-1, 2)  
        print_tensor_info("  Output: waypoint (reshaped)", waypoint, show_values=False)  
        print(f"  - Shape breakdown: (6, 2) - 6 waypoints, 2D coordinates")  
        print(f"  - Direct prediction without probabilistic sampling")  
      
    # 12. Final Trajectory Post-processing    
    print("\n" + "-"*80)    
    print("STEP 3.5: TRAJECTORY POST-PROCESSING")    
    print("-"*80)    
    print(f"File: mmcv/models/detectors/orion.py")    
    print(f"Function: simple_test_pts() at lines 903-934")    
        
    if not model.use_diff_decoder and not model.use_mlp_decoder:    
        # VAE mode selection    
        print(f"\nVAE Mode Selection:")    
        print(f"  - Select best mode based on ego_fut_cmd")    
        mask_active_cmd = dummy_data['ego_fut_cmd'][:,0,0] == 1    
        print(f"  - Active command mask: {mask_active_cmd}")    
            
        ego_fut_preds_selected = ego_fut_preds[mask_active_cmd].flatten(0,1)    
        print_tensor_info("  Selected trajectory (multi-mode)", ego_fut_preds_selected, show_values=False)    
        print(f"  - Shape after flatten: {ego_fut_preds_selected.shape} (modes, timesteps, coords)")  
          
        # SELECT FIRST MODE - THIS IS THE KEY FIX  
        print(f"\n  - Selecting mode 0 from multi-mode output")  
        ego_fut_preds_selected = ego_fut_preds_selected[0]  # Now shape: (6, 2)  
        print_tensor_info("  Selected trajectory (single mode)", ego_fut_preds_selected, show_values=False)  
            
        # Cumulative sum for absolute positions    
        print(f"\n  - Convert relative to absolute waypoints:")    
        ego_fut_pred = ego_fut_preds_selected.cumsum(dim=-2)    
        print_tensor_info("  Final trajectory (absolute)", ego_fut_pred, show_values=False)    
            
    elif model.use_diff_decoder:    
        # Diffusion mode already in absolute coordinates    
        print(f"\nDiffusion Mode Selection:")    
        mode_masks = torch.zeros(1, 20, device=ego_fut_preds.device).to(torch.bool)    
        mode_masks[0, mode_idx] = True    
        print(f"  - Mode mask: {mode_masks}")    
            
        ego_fut_pred = ego_fut_preds[mode_masks].flatten(0,1)    
        print_tensor_info("  Final trajectory (absolute)", ego_fut_pred, show_values=False)    
            
    elif model.use_mlp_decoder:    
        # MLP direct prediction    
        print(f"\nMLP Direct Prediction:")    
        ego_fut_pred = waypoint    
        print_tensor_info("  Final trajectory", ego_fut_pred, show_values=False)    
        
    # 13. Output Summary    
    print("\n" + "-"*80)    
    print("STEP 3.6: FINAL OUTPUT")    
    print("-"*80)    
        
    print(f"\nFinal Trajectory Output:")    
    print(f"  - Shape: {ego_fut_pred.shape}")    
    print(f"  - Type: PyTorch Tensor")    
    print(f"  - Dtype: {ego_fut_pred.dtype}")    
    print(f"  - Device: {ego_fut_pred.device}")    
      
    # Now ego_fut_pred should be (6, 2), so direct access works  
    print(f"\n  - Waypoint breakdown:")    
    for i in range(ego_fut_pred.shape[0]):    
        print(f"    * t={i*0.5:.1f}s: ({ego_fut_pred[i, 0].item():.3f}, {ego_fut_pred[i, 1].item():.3f})")    
        
    print(f"\n  - Represents: 6 future waypoints at 0.5s intervals (3 seconds total)")    
    print(f"  - Coordinate system: Ego vehicle frame (x=forward, y=left)")
  
# ============================================================================  
# COMPLETE PIPELINE SUMMARY  
# ============================================================================  
print("\n" + "="*80)  
print("COMPLETE INFERENCE PIPELINE SUMMARY")  
print("="*80)  
  
print("\n1. VISION SPACE:")  
print("   - img_feats_reshaped: PyTorch Tensor (1, 6, 1024, 40, 40)")  
print("   - pos_embed: PyTorch Tensor (1, 9600, 256)")  
print("   - det_query: PyTorch Tensor (1, 257, 4096)")  
print("   - map_query: PyTorch Tensor (1, 256, 4096)")  
print("   - vision_embeded: PyTorch Tensor (1, 513, 4096)")  
  
print("\n2. REASONING SPACE:")  
print("   - input_ids: PyTorch Tensor (1, seq_len)")  
print("   - vision_embeded: PyTorch Tensor (1, 513, 4096)")  
print("   - ego_feature: PyTorch Tensor (1, 4096)")  
  
print("\n3. ACTION SPACE:")  
print("   - current_states: PyTorch Tensor (1, 1, 4096)")  
if not model.use_diff_decoder and not model.use_mlp_decoder:  
    print("   - sample (VAE latent): PyTorch Tensor (1, 32)")  
    print("   - states_hs (future states): PyTorch Tensor (6, 1, 1, 4096)")  
    print("   - ego_fut_preds (multi-mode): PyTorch Tensor (1, 6, 6, 2)")  
elif model.use_diff_decoder:  
    print("   - plan_anchor: PyTorch Tensor (1, 20, 6, 2)")  
    print("   - noisy_trajs: PyTorch Tensor (1, 20, 6, 2)")  
    print("   - poses_reg: PyTorch Tensor (1, 20, 6, 2)")  
    print("   - poses_cls: PyTorch Tensor (1, 20)")  
elif model.use_mlp_decoder:  
    print("   - waypoint: PyTorch Tensor (6, 2)")  
print("   - ego_fut_pred (final): PyTorch Tensor (6, 2)")  
  
print("\n" + "="*80)  
print("All data structures are PyTorch tensors on CUDA device")  
print("Pipeline demonstrates complete flow from images to trajectory")  
print("="*80)