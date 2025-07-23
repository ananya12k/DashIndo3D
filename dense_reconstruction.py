import numpy as np
import cv2
import open3d as o3d
import pycolmap
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
import sys
import logging
import gc
from tqdm import tqdm
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class Config:
    BASE_PATH = Path("/scratch/Ananya_Kulkarni_AWR/MAP_LITE_IND")
    COLMAP_RECON_PATH = BASE_PATH / "colmap/d0/sparse/1"
    IMAGE_BASE_PATH = BASE_PATH / "images/d0"
    DEPTH_BASE_PATH = BASE_PATH / "depth_maps/d0"
    OUTPUT_PATH = BASE_PATH / "dense_reconstruction"
    DEPTH_EXTENSION = ".png"
    VOXEL_SIZE = 0.002
    MAX_POINTS_PER_IMAGE = 4000000
    MIN_DEPTH = 0.1
    MAX_DEPTH = 100.0
    MIN_POINTS_FOR_FITTING = 20
    BATCH_SIZE = 50
    STATS_SAMPLE_SIZE = 200

config = Config()

def get_camera_intrinsics(camera):
    if len(camera.params) >= 4: # Covers PINHOLE, OPENCV
        fx, fy, cx, cy = camera.params[:4]
        return fx, fy, cx, cy
    elif len(camera.params) == 3: # SIMPLE_PINHOLE
        fx, cx, cy = camera.params
        return fx, fx, cx, cy
    raise NotImplementedError(f"Unrecognized camera param length: {len(camera.params)}")

def get_image_pose(image):
    T_world_from_cam = image.cam_from_world().inverse()
    R, t = T_world_from_cam.rotation.matrix(), T_world_from_cam.translation
    return R.astype(np.float32), t.astype(np.float32)

def process_single_image(image, recon, global_min, global_max):
    try: R, t = get_image_pose(image)
    except: return None
    
    depth_path = config.DEPTH_BASE_PATH / Path(image.name).with_suffix(config.DEPTH_EXTENSION)
    if not depth_path.exists(): return None
    
    depth_mono = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_mono is None: return None
    
    depth_mono = depth_mono.astype(np.float32)
    if (global_max - global_min) < 1e-6: return None
    
    normalized_depth = 1.0 - ((depth_mono - global_min) / (global_max - global_min))

    mono_depths_for_fitting, metric_depths_for_fitting = [], []
    for p2d in image.points2D:
        if p2d.has_point3D():
            p_cam = R.T @ (recon.points3D[p2d.point3D_id].xyz - t)
            if p_cam[2] > config.MIN_DEPTH:
                u, v = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
                if 0 <= v < normalized_depth.shape[0] and 0 <= u < normalized_depth.shape[1] and normalized_depth[v, u] > 1e-5:
                    mono_depths_for_fitting.append(normalized_depth[v, u])
                    metric_depths_for_fitting.append(p_cam[2])

    if len(mono_depths_for_fitting) < config.MIN_POINTS_FOR_FITTING: return None
        
    try:
        ransac = RANSACRegressor(random_state=42).fit(np.array(mono_depths_for_fitting).reshape(-1, 1), np.array(metric_depths_for_fitting))
        scale, shift = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
        if not (0.01 < scale < 1000.0): return None
    except: return None
        
    scaled_depth = normalized_depth * scale + shift
    
    try: camera = recon.cameras[image.camera_id]; fx, fy, cx, cy = get_camera_intrinsics(camera)
    except: return None
        
    h, w = scaled_depth.shape
    subsample_factor = int(np.sqrt((h * w) / config.MAX_POINTS_PER_IMAGE)) + 1 if h * w > config.MAX_POINTS_PER_IMAGE else 1
    u, v = np.meshgrid(np.arange(0, w, subsample_factor), np.arange(0, h, subsample_factor))
    depth_sub = scaled_depth[::subsample_factor, ::subsample_factor]
    
    points_cam = np.stack([(u - cx) * depth_sub / fx, (v - cy) * depth_sub / fy, depth_sub], -1).reshape(-1, 3)
    points_world = (R @ points_cam.T).T + t
    
    color_image = cv2.imread(str(config.IMAGE_BASE_PATH / image.name))
    colors = (cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)[::subsample_factor, ::subsample_factor].reshape(-1, 3) / 255.0) if color_image is not None else np.full_like(points_world, 0.5)

    valid_mask = (depth_sub.flatten() > config.MIN_DEPTH) & (depth_sub.flatten() < config.MAX_DEPTH)
    return points_world[valid_mask].astype(np.float32), colors[valid_mask].astype(np.float32)

def downsample_and_combine(temp_clouds_dir, downsampled_clouds_dir):
    """
    CRITICAL FIX: Downsample each batch individually before combining.
    This avoids the massive memory peak that caused the 'Killed' error.
    """
    logger.info("Starting memory-safe downsampling and combination process...")
    downsampled_clouds_dir.mkdir(exist_ok=True)
    
    batch_files = sorted(temp_clouds_dir.glob("*.ply"))
    
    # Step 1: Downsample each large batch file and save it as a small file
    for batch_file in tqdm(batch_files, desc="Downsampling Batches"):
        pcd = o3d.io.read_point_cloud(str(batch_file))
        downsampled_pcd = pcd.voxel_down_sample(config.VOXEL_SIZE)
        
        downsampled_filepath = downsampled_clouds_dir / batch_file.name
        o3d.io.write_point_cloud(str(downsampled_filepath), downsampled_pcd)
        
        del pcd, downsampled_pcd
        gc.collect()

    # Step 2: Combine the small downsampled files
    logger.info("Combining downsampled batches...")
    final_pcd = o3d.geometry.PointCloud()
    for downsampled_file in tqdm(sorted(downsampled_clouds_dir.glob("*.ply")), desc="Combining Final Cloud"):
        final_pcd += o3d.io.read_point_cloud(str(downsampled_file))
        
    return final_pcd

def main():
    logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    config.OUTPUT_PATH.mkdir(exist_ok=True)
    
    # --- Create directories for temporary files ---
    temp_clouds_dir = config.OUTPUT_PATH / "temp_clouds"
    downsampled_clouds_dir = config.OUTPUT_PATH / "downsampled_temp_clouds"
    temp_clouds_dir.mkdir(exist_ok=True)
    
    # --- Check if batch processing is already complete ---
    if len(list(temp_clouds_dir.glob("*.ply"))) < 91:
        logger.info("Starting or resuming batch processing...")
        recon = pycolmap.Reconstruction(config.COLMAP_RECON_PATH)
        
        logger.info(f"Computing global depth statistics...")
        valid_depth_paths = [p for image in recon.images.values() if (p := config.DEPTH_BASE_PATH / Path(image.name).with_suffix(config.DEPTH_EXTENSION)).exists()]
        sample_paths = np.random.choice(valid_depth_paths, min(len(valid_depth_paths), config.STATS_SAMPLE_SIZE), replace=False)
        all_mins, all_maxs = [], []
        for path in tqdm(sample_paths, desc="Depth Stats"):
            depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            if depth is not None and depth.max() > 0:
                valid_depths = depth[depth > 0]
                if valid_depths.size > 0: all_mins.append(valid_depths.min()); all_maxs.append(valid_depths.max())
        global_min = np.percentile(all_mins, 5) if all_mins else 0
        global_max = np.percentile(all_maxs, 95) if all_maxs else 65535
        logger.info(f"Using global depth range for normalization: [{global_min:.2f}, {global_max:.2f}]")

        images_list = list(recon.images.items())
        total_batches = (len(images_list) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        
        for i in range(total_batches):
            batch_file_path = temp_clouds_dir / f"batch_{i:04d}.ply"
            if batch_file_path.exists():
                logger.info(f"Batch {i+1}/{total_batches} already exists. Skipping.")
                continue

            logger.info(f"--- Processing Batch {i+1}/{total_batches} ---")
            batch_images = images_list[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE]
            
            batch_points_list, batch_colors_list = [], []
            for image_id, image in tqdm(batch_images, desc=f"Batch {i+1}"):
                result = process_single_image(image, recon, global_min, global_max)
                if result:
                    points, colors = result
                    if points is not None and len(points) > 0:
                        batch_points_list.append(points)
                        batch_colors_list.append(colors)
            
            if batch_points_list:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.concatenate(batch_points_list))
                pcd.colors = o3d.utility.Vector3dVector(np.concatenate(batch_colors_list))
                o3d.io.write_point_cloud(str(batch_file_path), pcd)
                logger.info(f"Batch {i+1} successful with {len(pcd.points)} points.")
            else:
                logger.warning(f"Batch {i+1} generated no points.")
            gc.collect()
    else:
        logger.info("All batch files found. Skipping to final combination step.")

    # --- Final Combination Step (Memory Safe) ---
    final_pcd = downsample_and_combine(temp_clouds_dir, downsampled_clouds_dir)

    output_file = config.OUTPUT_PATH / "fused_dense_cloud.ply"
    o3d.io.write_point_cloud(str(output_file), final_pcd)
    logger.info(f"SUCCESS! Final point cloud has {len(final_pcd.points)} points. Saved to {output_file}")
    
    # --- Cleanup ---
    logger.info("Cleaning up temporary files...")
    for f in temp_clouds_dir.glob("*.ply"): f.unlink()
    temp_clouds_dir.rmdir()
    for f in downsampled_clouds_dir.glob("*.ply"): f.unlink()
    downsampled_clouds_dir.rmdir()

if __name__ == "__main__":
    main()
