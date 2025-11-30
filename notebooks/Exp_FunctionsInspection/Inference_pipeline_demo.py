import torch  
import numpy as np  

if not hasattr(np, 'Inf'):  
    np.Inf = np.inf

import matplotlib  
matplotlib.use('Agg')  # Use non-interactive backend  
import matplotlib.pyplot as plt
from mmcv import Config  
from mmcv.models import build_model  
from mmcv.utils import load_checkpoint  
from mmcv.datasets import build_dataset, build_dataloader  
from transformers import AutoTokenizer  
  
def extract_field(data_batch, key, default=None):  
    """Helper to extract field from batch, handling DataContainer or direct access"""  
    if key not in data_batch:  
        return default  
      
    value = data_batch[key]  
      
    if hasattr(value, 'data'):  
        return value.data[0]  
    elif isinstance(value, list):  
        return value[0]  
    else:  
        return value  
  
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
      
    if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:  
        print(f"Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.mean().item():.6f}")  
    else:  
        print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}")  
        print(f"Note: Mean not computed for integer dtype {tensor.dtype}")  
      
    print(f"{'='*60}\n")  
  
def display_input_images(img_tensor, save_path=None):  
    """Display the multi-view input images using direct tensor saving"""  
    try:  
        # Convert tensor to numpy for visualization  
        img_tensor = img_tensor.cpu()  
          
        # Handle different tensor formats  
        if img_tensor.dim() == 5:  # (B, N, C, H, W)  
            img_tensor = img_tensor[0]  # Remove batch dimension  
          
        if img_tensor.dim() == 4:  # (N, C, H, W)  
            num_views = img_tensor.shape[0]  
        elif img_tensor.dim() == 3:  # (C, H, W) - single view  
            num_views = 1  
            img_tensor = img_tensor.unsqueeze(0)  
        else:  
            print(f"Unexpected image tensor shape: {img_tensor.shape}")  
            return  
          
        # Use torchvision to save directly without matplotlib complications  
        import torchvision.utils as vutils  
          
        # Normalize images to [0, 1] range  
        img_normalized = img_tensor.clone()  
        if img_normalized.max() > 1.0:  
            img_normalized = img_normalized / 255.0  
        img_normalized = torch.clamp(img_normalized, 0, 1)  
          
        # Create a grid of images  
        if num_views <= 6:  
            grid = vutils.make_grid(img_normalized[:6], nrow=3, normalize=False, pad_value=1.0)  
        else:  
            grid = vutils.make_grid(img_normalized[:6], nrow=3, normalize=False, pad_value=1.0)  
          
        # Save using torchvision instead of matplotlib  
        if save_path:  
            vutils.save_image(grid, save_path, nrow=3)  
            print(f"Input images saved to: {save_path}")  
          
    except Exception as e:  
        print(f"Error in display_input_images: {e}")  
        return
 
def display_camera_calibration(cam_intrinsic, lidar2img):  
    """Display camera calibration matrices"""  
    print("\n" + "="*80)  
    print("CAMERA CALIBRATION DATA")  
    print("="*80)  
      
    print(f"\nCamera Intrinsic Matrices:")  
    print(f"  - Shape: {cam_intrinsic.shape}")  
    print(f"  - Number of cameras: {cam_intrinsic.shape[0]}")  
    print(f"  - Matrix size: {cam_intrinsic.shape[1]}x{cam_intrinsic.shape[2]}")  
      
    # Show first camera intrinsic matrix with proper indexing  
    print(f"\n  First camera intrinsic matrix:")  
    for i in range(3):  
        # Handle different tensor shapes  
        if cam_intrinsic.dim() == 3:  
            # Standard case: (N, 3, 3)  
            print(f"    Row {i}: [{cam_intrinsic[0, i, 0].item():.3f}, {cam_intrinsic[0, i, 1].item():.3f}, {cam_intrinsic[0, i, 2].item():.3f}]")  
        elif cam_intrinsic.dim() == 4:  
            # Batched case: (B, N, 3, 3)  
            print(f"    Row {i}: [{cam_intrinsic[0, 0, i, 0].item():.3f}, {cam_intrinsic[0, 0, i, 1].item():.3f}, {cam_intrinsic[0, 0, i, 2].item():.3f}]")  
        else:  
            print(f"    Unexpected tensor dimensions: {cam_intrinsic.dim()}")  
      
    print(f"\nLiDAR to Image Transformation Matrices:")  
    print(f"  - Shape: {lidar2img.shape}")  
    print(f"  - Number of cameras: {lidar2img.shape[0]}")  
    print(f"  - Matrix size: {lidar2img.shape[1]}x{lidar2img.shape[2]}")  
      
    # Show first lidar2img matrix with proper indexing  
    print(f"\n  First LiDAR to Image matrix:")  
    for i in range(3):  
        # Handle different tensor shapes  
        if lidar2img.dim() == 3:  
            # Standard case: (N, 4, 4)  
            print(f"    Row {i}: [{lidar2img[0, i, 0].item():.3f}, {lidar2img[0, i, 1].item():.3f}, {lidar2img[0, i, 2].item():.3f}, {lidar2img[0, i, 3].item():.3f}]")  
        elif lidar2img.dim() == 4:  
            # Batched case: (B, N, 4, 4)  
            print(f"    Row {i}: [{lidar2img[0, 0, i, 0].item():.3f}, {lidar2img[0, 0, i, 1].item():.3f}, {lidar2img[0, 0, i, 2].item():.3f}, {lidar2img[0, 0, i, 3].item():.3f}]")  
        else:  
            print(f"    Unexpected tensor dimensions: {lidar2img.dim()}")
  
def display_vehicle_state(can_bus):  
    """Display vehicle state information from CAN bus"""  
    print("\n" + "="*80)  
    print("VEHICLE STATE DATA (CAN BUS)")  
    print("="*80)  
      
    print(f"\nCAN Bus Data:")  
    print(f"  - Shape: {can_bus.shape}")  
    print(f"  - Total elements: {can_bus.shape[0]}")  
      
    # Based on ORION's can_bus format from the agent code  
    print(f"\nDecoded CAN Bus values:")  
    # Handle different tensor shapes  
    if can_bus.dim() == 1:  
        # 1D tensor - direct indexing  
        print(f"  - Position (x, y): ({can_bus[0].item():.3f}, {can_bus[1].item():.3f})")  
        print(f"  - Position Z: {can_bus[2].item():.3f} m")  
        print(f"  - Quaternion (w,x,y,z): [{can_bus[3].item():.3f}, {can_bus[4].item():.3f}, {can_bus[5].item():.3f}, {can_bus[6].item():.3f}]")  
        print(f"  - Speed: {can_bus[7].item():.3f} m/s")  
        print(f"  - Acceleration (x,y,z): [{can_bus[10].item():.3f}, {can_bus[11].item():.3f}, {can_bus[12].item():.3f}] m/s²")  
        print(f"  - Angular velocity (x,y,z): [{can_bus[13].item():.3f}, {can_bus[14].item():.3f}, {can_bus[15].item():.3f}] rad/s")  
        print(f"  - Heading angle: {can_bus[16].item():.3f} rad")  
        print(f"  - Heading angle (degrees): {can_bus[17].item():.3f}°")  
    elif can_bus.dim() == 2:  
        # Batched tensor - take first batch  
        print(f"  - Position (x, y): ({can_bus[0, 0].item():.3f}, {can_bus[0, 1].item():.3f})")  
        print(f"  - Position Z: {can_bus[0, 2].item():.3f} m")  
        print(f"  - Quaternion (w,x,y,z): [{can_bus[0, 3].item():.3f}, {can_bus[0, 4].item():.3f}, {can_bus[0, 5].item():.3f}, {can_bus[0, 6].item():.3f}]")  
        print(f"  - Speed: {can_bus[0, 7].item():.3f} m/s")  
        print(f"  - Acceleration (x,y,z): [{can_bus[0, 10].item():.3f}, {can_bus[0, 11].item():.3f}, {can_bus[0, 12].item():.3f}] m/s²")  
        print(f"  - Angular velocity (x,y,z): [{can_bus[0, 13].item():.3f}, {can_bus[0, 14].item():.3f}, {can_bus[0, 15].item():.3f}] rad/s")  
        print(f"  - Heading angle: {can_bus[0, 16].item():.3f} rad")  
        print(f"  - Heading angle (degrees): {can_bus[0, 17].item():.3f}°")  
    else:  
        print(f"  - Unexpected tensor dimensions: {can_bus.dim()}")

def display_navigation_commands(command, command_name=None):  
    """Display navigation command information"""  
    print("\n" + "="*80)  
    print("NAVIGATION COMMANDS")  
    print("="*80)  
      
    cmd_names = ['LEFT', 'RIGHT', 'STRAIGHT', 'LANE_FOLLOW', 'LANE_CHANGE_LEFT', 'LANE_CHANGE_RIGHT']  
      
    print(f"\nCurrent Command:")  
    print(f"  - Raw value: {command.item()}")  
    print(f"  - Command ID: {int(command.item())}")  # Convert to int  
    print(f"  - Command name: {cmd_names[int(command.item())] if 0 <= int(command.item()) < len(cmd_names) else 'Unknown'}")  # Fixed: convert to int  
      
    print(f"\nCommand Description:")  
    cmd_id = int(command.item())  
    if 0 <= cmd_id < len(cmd_names):  
        descriptions = {  
            0: "Turn left at the next intersection",  
            1: "Turn right at the next intersection",   
            2: "Continue straight ahead",  
            3: "Follow the current lane",  
            4: "Change lane to the left",  
            5: "Change lane to the right"  
        }  
        print(f"  - {descriptions.get(cmd_id, 'No description available')}")  
    else:  
        print(f"  - Unknown command (ID: {cmd_id})")
  
def display_temporal_data(timestamp):  
    """Display temporal information"""  
    print("\n" + "="*80)  
    print("TEMPORAL DATA")  
    print("="*80)  
      
    print(f"\nTimestamp Information:")  
    print(f"  - Timestamp value: {timestamp.item():.3f}")  # Fixed: added .item()  
    print(f"  - Estimated time: {timestamp.item():.1f} seconds")  # Fixed: added .item()  
    print(f"  - Frame rate: ~20 Hz (assuming 0.05s per frame)") 
  
def display_ego_poses(ego_pose, ego_pose_inv):  
    """Display ego vehicle pose information"""  
    print("\n" + "="*80)  
    print("EGO VEHICLE POSE")  
    print("="*80)  
    print(f"\nEgo Pose (World to Ego):")  
    print(f" - Shape: {ego_pose.shape}")  
      
    # Handle different tensor shapes properly  
    if ego_pose.dim() == 2 and ego_pose.shape == (4, 4):  
        # Standard 4x4 matrix case  
        print(f" - Translation (x,y,z): [{ego_pose[0, 3].item():.3f}, {ego_pose[1, 3].item():.3f}, {ego_pose[2, 3].item():.3f}]")  
        print(f" - Rotation matrix (first 3x3):")  
        for i in range(3):  
            print(f" Row {i}: [{ego_pose[i, 0].item():.3f}, {ego_pose[i, 1].item():.3f}, {ego_pose[i, 2].item():.3f}]")  
    elif ego_pose.dim() == 3:  
        # Batched case: (B, 4, 4) or (N, 4, 4)  
        if ego_pose.shape[0] == 1:  
            # Single batch, remove batch dimension  
            ego_pose_squeezed = ego_pose.squeeze(0)  
            print(f" - Translation (x,y,z): [{ego_pose_squeezed[0, 3].item():.3f}, {ego_pose_squeezed[1, 3].item():.3f}, {ego_pose_squeezed[2, 3].item():.3f}]")  
            print(f" - Rotation matrix (first 3x3):")  
            for i in range(3):  
                print(f" Row {i}: [{ego_pose_squeezed[i, 0].item():.3f}, {ego_pose_squeezed[i, 1].item():.3f}, {ego_pose_squeezed[i, 2].item():.3f}]")  
        else:  
            # Multiple batches, take first one  
            print(f" - Translation (x,y,z): [{ego_pose[0, 0, 3].item():.3f}, {ego_pose[0, 1, 3].item():.3f}, {ego_pose[0, 2, 3].item():.3f}]")  
            print(f" - Rotation matrix (first 3x3):")  
            for i in range(3):  
                print(f" Row {i}: [{ego_pose[0, i, 0].item():.3f}, {ego_pose[0, i, 1].item():.3f}, {ego_pose[0, i, 2].item():.3f}]")  
    elif ego_pose.dim() == 4:  
        # Fully batched case: (B, N, 4, 4)  
        print(f" - Translation (x,y,z): [{ego_pose[0, 0, 0, 3].item():.3f}, {ego_pose[0, 0, 1, 3].item():.3f}, {ego_pose[0, 0, 2, 3].item():.3f}]")  
        print(f" - Rotation matrix (first 3x3):")  
        for i in range(3):  
            print(f" Row {i}: [{ego_pose[0, 0, i, 0].item():.3f}, {ego_pose[0, 0, i, 1].item():.3f}, {ego_pose[0, 0, i, 2].item():.3f}]")  
    else:  
        print(f" - Unexpected tensor dimensions: {ego_pose.dim()}, shape: {ego_pose.shape}")  
        # Try to extract translation by flattening the last dimension  
        if ego_pose.numel() >= 16:  
            flat_pose = ego_pose.flatten()  
            print(f" - Attempted translation (x,y,z): [{flat_pose[3].item():.3f}, {flat_pose[7].item():.3f}, {flat_pose[11].item():.3f}]")  
      
    print(f"\nEgo Pose Inverse (Ego to World):")  
    print(f" - Shape: {ego_pose_inv.shape}")  
      
    # Handle inverse pose with same logic  
    if ego_pose_inv.dim() == 2 and ego_pose_inv.shape == (4, 4):  
        print(f" - Translation (x,y,z): [{ego_pose_inv[0, 3].item():.3f}, {ego_pose_inv[1, 3].item():.3f}, {ego_pose_inv[2, 3].item():.3f}]")  
        print(f" - Rotation matrix (first 3x3):")  
        for i in range(3):  
            print(f" Row {i}: [{ego_pose_inv[i, 0].item():.3f}, {ego_pose_inv[i, 1].item():.3f}, {ego_pose_inv[i, 2].item():.3f}]")  
    elif ego_pose_inv.dim() == 3:  
        if ego_pose_inv.shape[0] == 1:  
            ego_pose_inv_squeezed = ego_pose_inv.squeeze(0)  
            print(f" - Translation (x,y,z): [{ego_pose_inv_squeezed[0, 3].item():.3f}, {ego_pose_inv_squeezed[1, 3].item():.3f}, {ego_pose_inv_squeezed[2, 3].item():.3f}]")  
            print(f" - Rotation matrix (first 3x3):")  
            for i in range(3):  
                print(f" Row {i}: [{ego_pose_inv_squeezed[i, 0].item():.3f}, {ego_pose_inv_squeezed[i, 1].item():.3f}, {ego_pose_inv_squeezed[i, 2].item():.3f}]")  
        else:  
            print(f" - Translation (x,y,z): [{ego_pose_inv[0, 0, 3].item():.3f}, {ego_pose_inv[0, 1, 3].item():.3f}, {ego_pose_inv[0, 2, 3].item():.3f}]")  
            print(f" - Rotation matrix (first 3x3):")  
            for i in range(3):  
                print(f" Row {i}: [{ego_pose_inv[0, i, 0].item():.3f}, {ego_pose_inv[0, i, 1].item():.3f}, {ego_pose_inv[0, i, 2].item():.3f}]")  
    elif ego_pose_inv.dim() == 4:  
        print(f" - Translation (x,y,z): [{ego_pose_inv[0, 0, 0, 3].item():.3f}, {ego_pose_inv[0, 0, 1, 3].item():.3f}, {ego_pose_inv[0, 0, 2, 3].item():.3f}]")  
        print(f" - Rotation matrix (first 3x3):")  
        for i in range(3):  
            print(f" Row {i}: [{ego_pose_inv[0, 0, i, 0].item():.3f}, {ego_pose_inv[0, 0, i, 1].item():.3f}, {ego_pose_inv[0, 0, i, 2].item():.3f}]")  
    else:  
        print(f" - Unexpected tensor dimensions: {ego_pose_inv.dim()}, shape: {ego_pose_inv.shape}")  
        if ego_pose_inv.numel() >= 16:  
            flat_pose_inv = ego_pose_inv.flatten()  
            print(f" - Attempted translation (x,y,z): [{flat_pose_inv[3].item():.3f}, {flat_pose_inv[7].item():.3f}, {flat_pose_inv[11].item():.3f}]")
  
def display_all_inputs(real_data):  
    """Display all input data types"""  
    # Display images (already implemented)  
    display_input_images(real_data['img'], save_path='input_images.png')  
      
    # Display camera calibration  
    display_camera_calibration(  
        real_data['cam_intrinsic'].cpu(),   
        real_data['lidar2img'].cpu()  
    )  
      
    # Display vehicle state  
    display_vehicle_state(real_data['can_bus'].cpu())  
      
    # Display navigation commands  
    display_navigation_commands(  
        real_data['command'].cpu(),   
        real_data['ego_fut_cmd'].cpu()  
    )  
      
    # Display temporal data  
    display_temporal_data(real_data['timestamp'].cpu())  
      
    # Display ego poses  
    display_ego_poses(  
        real_data['ego_pose'].cpu(),   
        real_data['ego_pose_inv'].cpu()  
    )

# Load config  
cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_agent.py')  
  
# Build model  
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))  
checkpoint = load_checkpoint(model, 'ckpts/Orion.pth', map_location='cpu')  
model.cuda()  
model.eval()  
  
# Load tokenizer  
tokenizer = AutoTokenizer.from_pretrained('ckpts/pretrain_qformer/', trust_remote_code=True)  
  
print("="*80)  
print("ORION MANUAL COMPONENT TRACING - VISION, REASONING & ACTION SPACES")  
print("USING REAL BENCH2DRIVE DATASET")  
print("="*80)  
  
# Build dataset  
dataset = build_dataset(cfg.data.test)  
print(f"\nDataset loaded: {len(dataset)} samples")  
  
# Build dataloader  
data_loader = build_dataloader(  
    dataset,  
    samples_per_gpu=1,  
    workers_per_gpu=0,  
    dist=False,  
    shuffle=False  
)  
  
# Get first batch  
data_iter = iter(data_loader)  
data_batch = next(data_iter)  
  
print(f"\nBatch keys: {data_batch.keys()}")  

# === ADD THIS SECTION TO SHOW SAMPLE INPUT DATA ===    
print("\n" + "="*80)    
print("SAMPLE INPUT DATA INSPECTION")    
print("="*80)    
    
# Show raw image data info    
if 'img' in data_batch:    
    img_data = extract_field(data_batch, 'img')    
    print(f"\nRaw Image Data:")    
    print(f"  - Type: {type(img_data)}")    
    print(f"  - Shape: {img_data.shape if hasattr(img_data, 'shape') else 'N/A'}")    
    print(f"  - Dtype: {img_data.dtype if hasattr(img_data, 'dtype') else 'N/A'}")    
    if hasattr(img_data, 'shape') and len(img_data.shape) == 4:    
        print(f"  - Sample pixel values (first image, center pixel):")    
        print(f"    RGB: {img_data[0, :, img_data.shape[2]//2, img_data.shape[3]//2]}")    
    
# Show command data    
if 'command' in data_batch:    
    cmd_data = extract_field(data_batch, 'command')    
    print(f"\nCommand Data:")    
    print(f"  - Value: {cmd_data.item() if hasattr(cmd_data, 'item') and cmd_data.numel() == 1 else cmd_data}")    
    print(f"  - Type: {type(cmd_data)}")    
    
# Show can_bus data    
if 'can_bus' in data_batch:    
    can_bus_data = extract_field(data_batch, 'can_bus')    
    print(f"\nCAN Bus Data (vehicle state):")    
    print(f"  - Shape: {can_bus_data.shape if hasattr(can_bus_data, 'shape') else 'N/A'}")    
    if hasattr(can_bus_data, 'shape') and len(can_bus_data.shape) >= 1:    
        # Handle both 1D and batched tensors  
        if can_bus_data.dim() == 1:  
            # 1D tensor - direct indexing  
            print(f"  - Position (x, y): ({can_bus_data[0].item():.3f}, {can_bus_data[1].item():.3f})")    
            print(f"  - Speed: {can_bus_data[7].item():.3f} m/s")    
            print(f"  - Heading: {can_bus_data[16].item():.3f} rad")    
        elif can_bus_data.dim() == 2:  
            # Batched tensor - take first batch  
            print(f"  - Position (x, y): ({can_bus_data[0, 0].item():.3f}, {can_bus_data[0, 1].item():.3f})")    
            print(f"  - Speed: {can_bus_data[0, 7].item():.3f} m/s")    
            print(f"  - Heading: {can_bus_data[0, 16].item():.3f} rad")    
        else:  
            print(f"  - Unexpected tensor dimensions: {can_bus_data.dim()}")    
    
print("="*80)    
# === END OF SAMPLE DATA INSPECTION ===
  
# Handle img_metas  
if isinstance(data_batch['img_metas'], list):  
    if isinstance(data_batch['img_metas'][0], list):  
        real_img_metas = data_batch['img_metas'][0]  
    else:  
        real_img_metas = data_batch['img_metas']  
else:  
    real_img_metas = data_batch['img_metas'].data[0]  
  
print(f"\nSample Information:")  
print(f"  - Scene token: {real_img_metas[0].get('scene_token', 'N/A')}")  
print(f"  - Frame index: {real_img_metas[0].get('frame_idx', 'N/A')}")  
  
# Extract data  
real_data = {  
    'img': extract_field(data_batch, 'img').cuda(),  
    'img_feats': None,  
    'cam_intrinsic': extract_field(data_batch, 'cam_intrinsic').cuda(),  
    'lidar2img': extract_field(data_batch, 'lidar2img').cuda(),  
    'can_bus': extract_field(data_batch, 'can_bus').cuda(),  
    'command': extract_field(data_batch, 'command').cuda(),  
    'ego_fut_cmd': extract_field(data_batch, 'ego_fut_cmd').cuda(),  
    'timestamp': extract_field(data_batch, 'timestamp').cuda(),  
    'ego_pose': extract_field(data_batch, 'ego_pose').cuda(),  
    'ego_pose_inv': extract_field(data_batch, 'ego_pose_inv').cuda(),  
}  
  
print(f"\n✓ Real data loaded successfully")  
print(f"  - Image shape: {real_data['img'].shape}")  
print(f"  - Command: {real_data['command']}")  

# Display all input data  
print("\n" + "="*80)  
print("DISPLAYING ALL INPUT DATA")  
print("="*80)  
display_all_inputs(real_data) 
 

with torch.no_grad():  
    # ========================================================================  
    # PART 1: VISION SPACE  
    # ========================================================================  
    print("\n" + "="*80)  
    print("PART 1: VISION SPACE (REAL DATA)")  
    print("="*80)  
      
    # 1. Feature Extraction  
    print("\n" + "-"*80)  
    print("STEP 1.1: FEATURE EXTRACTION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: extract_img_feat() at lines 331-359")  
      
    print_tensor_info("Input: img (REAL)", real_data['img'], show_values=False)  
      
    img_feats = model.extract_img_feat(real_data['img'])  
      
    print_tensor_info("Output: img_feats_reshaped (REAL)", img_feats, show_values=False)  
    print(f"✓ Feature extraction complete")  
      
    # 2. Position Encoding  
    print("\n" + "-"*80)  
    print("STEP 1.2: POSITION ENCODING")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: position_embeding() at lines 381-418")  
      
    real_data['img_feats'] = img_feats  
    location = model.prepare_location(real_img_metas, **real_data)  
      
    print_tensor_info("Input: location (REAL)", location, show_values=False)  
      
    pos_embed = model.position_embeding(real_data, location, real_img_metas)  
      
    print_tensor_info("Output: pos_embed (REAL)", pos_embed, show_values=False)  
    print(f"✓ Position encoding complete")  
      
    # 3. Object Detection Head  
    print("\n" + "-"*80)  
    print("STEP 1.3: OBJECT DETECTION HEAD")  
    print("-"*80)  
    print(f"File: mmcv/models/dense_heads/orion_head.py")  
    print(f"Function: forward() at lines 709-834")  
      
    outs_bbox, det_query = model.pts_bbox_head(real_img_metas, pos_embed, **real_data)  
      
    print_tensor_info("Output: det_query (REAL)", det_query, show_values=False)  
    print(f"✓ Object detection complete")  
    print(f"  - Tokens 0-255: 256 object queries")  
    print(f"  - Token 256: 1 planning token")  
      
    # 4. Map Detection Head  
    print("\n" + "-"*80)  
    print("STEP 1.4: MAP DETECTION HEAD")  
    print("-"*80)  
    print(f"File: mmcv/models/dense_heads/orion_head_map.py")  
    print(f"Function: forward() at lines 388-484")  
      
    outs_lane, map_query = model.map_head(real_img_metas, pos_embed, **real_data)  
      
    print_tensor_info("Output: map_query (REAL)", map_query, show_values=False)  
    print(f"✓ Map detection complete")  
      
    # 5. Vision Token Concatenation  
    print("\n" + "-"*80)  
    print("STEP 1.5: VISION TOKEN CONCATENATION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: torch.cat() at line 767")  
      
    vision_embeded = torch.cat([det_query, map_query], dim=1)  
      
    print_tensor_info("Output: vision_embeded (REAL)", vision_embeded, show_values=False)  
    print(f"✓ Concatenation complete: 513 tokens (257 object + 256 map)")  
      
    # ========================================================================  
    # PART 2: REASONING SPACE  
    # ========================================================================  
    print("\n" + "="*80)  
    print("PART 2: REASONING SPACE (REAL DATA)")  
    print("="*80)  
      
    # 6. Text Input Preparation  
    print("\n" + "-"*80)  
    print("STEP 2.1: TEXT INPUT PREPARATION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: Tokenization in simple_test_pts() at lines 768-769")  
      
    # Handle input_ids  
    if 'input_ids' in data_batch:  
        input_ids_raw = data_batch['input_ids']  
          
        if isinstance(input_ids_raw, list):  
            if len(input_ids_raw) > 0 and isinstance(input_ids_raw[0], list):  
                input_ids_cuda = []  
                for conversation_round in input_ids_raw[0]:  
                    if isinstance(conversation_round, torch.Tensor):  
                        input_ids_cuda.append(conversation_round.cuda())  
                    else:  
                        input_ids_cuda.append(conversation_round)  
            else:  
                input_ids_cuda = [t.cuda() if isinstance(t, torch.Tensor) else t for t in input_ids_raw]  
              
            print(f"\nInput IDs structure:")  
            print(f"  - Type: List of conversation rounds")  
            print(f"  - Number of rounds: {len(input_ids_cuda)}")  
              
            if len(input_ids_cuda) > 0:  
                first_round = input_ids_cuda[0]  
                  
                if isinstance(first_round, torch.Tensor):  
                    input_ids = first_round  
                elif isinstance(first_round, list) and len(first_round) > 0:  
                    input_ids = first_round[0] if isinstance(first_round[0], torch.Tensor) else torch.stack(first_round)  
                else:  
                    print("\nWarning: input_ids structure is unexpected")  
                    input_ids = None  
                  
                if input_ids is not None:  
                    print_tensor_info("Tokenized input_ids (REAL) - First Round", input_ids, show_values=True, max_elements=20)  
            else:  
                print("\nNo input_ids available in batch")  
                input_ids = None  
        else:  
            print(f"\nUnexpected input_ids type: {type(input_ids_raw)}")  
            input_ids = input_ids_raw  
    else:  
        print("\nNo input_ids found in data_batch")  
        input_ids = None  
      
    # 7. LLM Inference for Planning Token  
    print("\n" + "-"*80)  
    print("STEP 2.2: LLM INFERENCE FOR PLANNING TOKEN")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: lm_head.inference_ego() at lines 783-792")  
      
    # Ensure input_ids has batch dimension  
    if input_ids is not None and input_ids.dim() == 1:  
        input_ids = input_ids.unsqueeze(0)  
      
    print(f"\nInputs to LLM (REAL DATA):")  
    print(f"  - inputs (text tokens): {input_ids.shape}")  
    print(f"  - images (vision tokens): {vision_embeded.shape}")  
    print(f"  - return_ego_feature: True")  
      
    print(f"\n⚠ Running ACTUAL LLM inference (this may take time)...")  
    ego_feature = model.lm_head.inference_ego(  
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
      
    print_tensor_info("Output: ego_feature (REAL from LLM)", ego_feature, show_values=False)  
    print(f"✓ Planning token extraction complete")  
      
    # ========================================================================  
    # PART 3: ACTION SPACE  
    # ========================================================================  
    print("\n" + "="*80)  
    print("PART 3: ACTION SPACE (REAL DATA)")  
    print("="*80)  
      
    # 8. Planning Token Preparation  
    print("\n" + "-"*80)  
    print("STEP 3.1: PLANNING TOKEN PREPARATION")  
    print("-"*80)  
    print(f"File: mmcv/models/detectors/orion.py")  
    print(f"Function: simple_test_pts() at lines 794-795")  
      
    ego_feature = ego_feature.to(torch.float32)  
    current_states = ego_feature.unsqueeze(1)  
      
    print_tensor_info("Input: ego_feature (REAL)", ego_feature, show_values=False)  
    print_tensor_info("Output: current_states (REAL)", current_states, show_values=False)  
    print(f"✓ Planning token prepared")  
      
    # 9. Trajectory Generation - VAE Mode    
    print("\n" + "-"*80)    
    print("STEP 3.2: TRAJECTORY GENERATION (VAE MODE)")    
    print("-"*80)    
    print(f"File: mmcv/models/detectors/orion.py")    
    print(f"Function: VAE-based planning at lines 796-819")    
        
    if not model.use_diff_decoder and not model.use_mlp_decoder:    
        print(f"\nVAE Planning Pipeline:")    
          
        B = 1  # Batch size  
          
        # 3.2.1: Distribution Sampling    
        print(f"\n  3.2.1: Distribution Sampling")    
        print(f"  Function: distribution_forward() at lines 801-804")    
        print(f"  File: mmcv/models/detectors/orion.py")    
          
        # Call actual distribution_forward  
        sample, output_distribution = model.distribution_forward(  
            current_states, None, None  
        )  
        print_tensor_info("  Output: sample (latent code)", sample, show_values=False)    
        print(f"  ✓ Sampled 32-dim latent code from present distribution")    
            
        # 3.2.2: Future State Prediction    
        print(f"\n  3.2.2: Future State Prediction")    
        print(f"  Function: future_states_predict() at lines 808-810")    
        print(f"  File: mmcv/models/detectors/orion.py")    
          
        # Call actual future_states_predict  
        hidden_states = ego_feature.unsqueeze(1)    
        states_hs, future_states_hs = model.future_states_predict(  
            B, sample, hidden_states, current_states  
        )  
          
        print_tensor_info("  Input: hidden_states", hidden_states, show_values=False)    
        print_tensor_info("  Output: states_hs (future states)", states_hs, show_values=False)    
        print(f"  ✓ Predicted future states for 6 timesteps")    
            
        # 3.2.3: Trajectory Decoding    
        print(f"\n  3.2.3: Trajectory Decoding")    
        print(f"  Function: ego_fut_decoder() at lines 814-817")    
        print(f"  File: mmcv/models/detectors/orion.py")    
          
        # Call actual ego_fut_decoder  
        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)    
        ego_fut_trajs_list = []    
        for i in range(6):    
            outputs_ego_trajs = model.ego_fut_decoder(ego_query_hs[i]).reshape(B, model.ego_fut_mode, 2)  
            ego_fut_trajs_list.append(outputs_ego_trajs)    
            if i == 0:    
                print_tensor_info(f"  Output timestep {i}: ego_trajs", outputs_ego_trajs, show_values=False)    
            
        ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)  # (1, 6, 6, 2)    
        print_tensor_info("  Final: ego_fut_preds (all modes & timesteps)", ego_fut_preds, show_values=False)    
        print(f"  ✓ Generated 6 trajectory modes, each with 6 waypoints")    
            
    # 10. Trajectory Post-processing        
    print("\n" + "-"*80)        
    print("STEP 3.3: TRAJECTORY POST-PROCESSING")        
    print("-"*80)        
    print(f"File: mmcv/models/detectors/orion.py")        
    print(f"Function: simple_test_pts() at lines 903-934")        
            
    if not model.use_diff_decoder and not model.use_mlp_decoder:        
        print(f"\nVAE Mode Selection:")        
            
        # Mode selection based on ego_fut_cmd        
        mask_active_cmd = real_data['ego_fut_cmd'][:, 0, 0] == 1        
        print(f"  - Active command mask: {mask_active_cmd}")        
            
        # Apply mask and flatten - this mimics lines 904-906    
        ego_fut_preds_masked = ego_fut_preds[mask_active_cmd]  # Shape: (1, 6, 6, 2) if mask is [True]    
        print_tensor_info("  After masking", ego_fut_preds_masked, show_values=False)        
          
        # Check if we have valid predictions after masking  
        if ego_fut_preds_masked.numel() == 0:  
            print("  Warning: No active commands found, using zero trajectory")  
            ego_fut_pred = torch.zeros(6, 2).cuda()  
        else:  
            # Flatten batch and mode dimensions: (1, 6, 6, 2) -> (6, 6, 2)    
            ego_fut_preds_flattened = ego_fut_preds_masked.flatten(0, 1)  # (6, 6, 2)    
            print_tensor_info("  After flatten(0,1)", ego_fut_preds_flattened, show_values=False)    
              
            # Select first mode: (6, 6, 2) -> (6, 2)    
            # This represents selecting mode 0 from the 6 modes    
            ego_fut_pred = ego_fut_preds_flattened[0:6]  # Take first 6 waypoints to ensure 2D shape  
            if ego_fut_pred.dim() == 1:  
                ego_fut_pred = ego_fut_pred.view(6, 2)  # Ensure 2D shape  
            print_tensor_info("  Selected mode trajectory", ego_fut_pred, show_values=False)        
            
        # Apply cumulative sum to convert relative to absolute coordinates        
        ego_fut_pred = ego_fut_pred.cumsum(dim=0)  # Use dim=0 instead of dim=-2 for 2D tensor        
        print_tensor_info("  Final trajectory (cumsum)", ego_fut_pred, show_values=False)        
        print(f"  ✓ Converted relative waypoints to absolute positions")
            
    elif model.use_diff_decoder:    
        print(f"\nDiffusion Mode Selection:")    
        # For diffusion, would need to run actual diffusion decoder  
        print(f"  ⚠ Diffusion mode not fully implemented in manual trace")  
        ego_fut_pred = torch.zeros(6, 2).cuda()    
        print_tensor_info("  Final trajectory (placeholder)", ego_fut_pred, show_values=False)    
            
    elif model.use_mlp_decoder:    
        print(f"\nMLP Direct Prediction:")    
        # For MLP, would call waypoint_decoder  
        print(f"  ⚠ MLP mode not fully implemented in manual trace")  
        ego_fut_pred = torch.zeros(6, 2).cuda()    
        print_tensor_info("  Final trajectory (placeholder)", ego_fut_pred, show_values=False)
      
    # 11. Final Output Display  
    print("\n" + "-"*80)  
    print("STEP 3.4: FINAL OUTPUT")  
    print("-"*80)  
      
    print(f"\nFinal Trajectory Output:")  
    print_tensor_info("ego_fut_pred", ego_fut_pred, show_values=True, max_elements=12)  
      
    print(f"\n  - Waypoint breakdown:")  
    for i in range(6):  
        print(f"    * t={i*0.5:.1f}s: ({ego_fut_pred[i, 0].item():.3f}, {ego_fut_pred[i, 1].item():.3f})")  
      
    print(f"\n  - Represents: 6 future waypoints at 0.5s intervals (3 seconds total)")  
    print(f"  - Coordinate system: Ego vehicle frame (x=forward, y=left)")  
      
    # Summary  
    print("\n" + "="*80)  
    print("SUMMARY: DATA FLOW THROUGH ALL SPACES")  
    print("="*80)  
      
    print("\n1. VISION SPACE (REAL DATA):")  
    print("   - img: PyTorch Tensor (1, 6, 3, 640, 640)")  
    print("   - img_feats_reshaped: PyTorch Tensor (1, 6, 1024, 40, 40)")  
    print("   - pos_embed: PyTorch Tensor (1, 9600, 256)")  
    print("   - det_query: PyTorch Tensor (1, 257, 4096)")  
    print("   - map_query: PyTorch Tensor (1, 256, 4096)")  
    print("   - vision_embeded: PyTorch Tensor (1, 513, 4096)")  
      
    print("\n2. REASONING SPACE (REAL DATA):")  
    print("   - input_ids: PyTorch Tensor (1, seq_len)")  
    print("   - vision_embeded: PyTorch Tensor (1, 513, 4096)")  
    print("   - ego_feature: PyTorch Tensor (1, 4096)")  
      
    print("\n3. ACTION SPACE (REAL DATA):")  
    print("   - current_states: PyTorch Tensor (1, 1, 4096)")  
    if not model.use_diff_decoder and not model.use_mlp_decoder:  
        print("   - sample (VAE latent): PyTorch Tensor (1, 32)")  
        print("   - states_hs (future states): PyTorch Tensor (6, 1, 1, 4096)")  
        print("   - ego_fut_preds (multi-mode): PyTorch Tensor (1, 6, 6, 2)")  
    print("   - ego_fut_pred (final): PyTorch Tensor (6, 2)")  
      
    print("\n4. RESULTS:")  
    print(f"   - Trajectory: 6 waypoints over 3 seconds")  
    print(f"   - Planning mechanism: {'VAE' if not model.use_diff_decoder and not model.use_mlp_decoder else 'Diffusion' if model.use_diff_decoder else 'MLP'}")  
      
    print("\n" + "="*80)  
    print("All data processed from REAL Bench2Drive dataset sample")  
    print("Pipeline demonstrates complete flow from real images to trajectory")  
    print("="*80)  
  