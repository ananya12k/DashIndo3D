#!/bin/bash
#SBATCH -A inai
#SBATCH -c 36
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=logs/pose_estimation.txt
#SBATCH --nodelist=gnode098
#SBATCH --job-name=pose_estimation

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

echo "Job started on $(hostname) at $(date)"

SEQUENCES=("d0" "d1" "d2")
DATA_ROOT="/scratch/Ananya_Kulkarni_AWR/MAP_LITE_IND"

for SEQ in "${SEQUENCES[@]}"; do
  echo "=== Processing $SEQ ==="
  IMG_DIR="${DATA_ROOT}/images/${SEQ}"
  OUT_DIR="${DATA_ROOT}/colmap/${SEQ}"
  DB_PATH="${OUT_DIR}/database.db"
  SPARSE_DIR="${OUT_DIR}/sparse"

  mkdir -p "$SPARSE_DIR" "$OUT_DIR"

  echo "--- Feature Extraction (CPU) ---"
  colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMG_DIR" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0

  echo "--- Sequential Matching (GPU) ---"
  colmap sequential_matcher \
    --database_path "$DB_PATH" \
    --SiftMatching.use_gpu 1

  echo "--- Sparse Mapping ---"
  colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMG_DIR" \
    --output_path "$SPARSE_DIR"
done

echo "Pose estimation done for all sequences at $(date)"
