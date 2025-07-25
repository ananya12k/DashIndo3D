import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

# -------- Correct COLMAP binary writing function -------- #
def write_colmap_depth_map(path, depth_map):
    """
    Writes a depth map to the specific COLMAP-compatible .bin format.
    
    Args:
        path (str): Path to the output .bin file.
        depth_map (np.ndarray): HxW depth map in metric units (meters).
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert to float32
    depth_map = depth_map.astype(np.float32)
    
    with open(path, "wb") as fid:
        # Write header in COLMAP's expected format
        width, height = depth_map.shape[1], depth_map.shape[0]
        num_bytes = width * height * 4  # float32 is 4 bytes
        
        fid.write(b"&" + np.array([width], dtype=np.int32).tobytes())
        fid.write(b"&" + np.array([height], dtype=np.int32).tobytes())
        fid.write(b"&" + np.array([num_bytes], dtype=np.int64).tobytes())
        fid.write(b"&")

        # Write data
        depth_map.tobytes(order='C')

# -------- Modified Convert function (runs in parallel) -------- #
def process_one_file(args):
    input_file, output_file, scale = args
    try:
        # Correctly read 16-bit grayscale image
        depth_16bit = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
        if depth_16bit is None:
            print(f"[ERROR] Could not read: {input_file}")
            return

        # Scale from mm (integer) to meters (float)
        metric_depth = depth_16bit.astype(np.float32) * scale
        
        # Use the corrected writing function
        write_colmap_depth_map(output_file, metric_depth)

        print(f"[OK] {os.path.basename(input_file)} → {output_file}")
    except Exception as e:
        print(f"[FAIL] {input_file}: {e}")

# -------- Modified file collection and pool execution -------- #
def process_all_depth_maps_parallel(depth_root_dir, image_root_dir, scale=0.001, num_workers=8):
    tasks = []

    # Loop through sequence directories like 'd0', 'd1', etc.
    for seq_name in sorted(os.listdir(depth_root_dir)):
        depth_seq_dir = os.path.join(depth_root_dir, seq_name)
        image_seq_dir = os.path.join(image_root_dir, seq_name)

        if not os.path.isdir(depth_seq_dir) or not os.path.isdir(image_seq_dir):
            continue

        print(f"--- Processing sequence: {seq_name} ---")

        for file in os.listdir(depth_seq_dir):
            if file.endswith('.png'):
                input_file = os.path.join(depth_seq_dir, file)
                
                # ---- CRITICAL FIX 2: Correct output path and name ----
                # Output file goes into the IMAGE directory with the correct name.
                output_file = os.path.join(image_seq_dir, file + '.geometric.bin')
                
                tasks.append((input_file, output_file, scale))

    if not tasks:
        print("No tasks found. Check your directory paths.")
        return
        
    print(f"Total files to convert: {len(tasks)}")

    # Multiprocessing pool
    with Pool(processes=num_workers) as pool:
        pool.map(process_one_file, tasks)

if __name__ == "__main__":
    # Define your root directories
    ROOT_DIR = "/scratch/Ananya_Kulkarni_AWR/MAP_LITE_IND"
    depth_root_dir = os.path.join(ROOT_DIR, "depth_16bit") # Where your 16-bit PNGs are
    image_root_dir = os.path.join(ROOT_DIR, "images")      # Where your RGB images are

    num_cpu = min(42, cpu_count())
    process_all_depth_maps_parallel(depth_root_dir, image_root_dir, scale=0.001, num_workers=num_cpu)