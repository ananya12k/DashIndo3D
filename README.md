# DashIndo3D: Lightweight Local Map Reconstruction from Indian Monocular Dashcam Videos
**A monocular 3D reconstruction pipeline tailored for unstructured road environments using the IDD-Multimodal dataset**

##  Overview
DashIndo3D aims to reconstruct lightweight, local 3D maps from monocular dashcam videos captured in Indian traffic conditions. The project explores whether autonomous navigation systems can estimate 3D scene structure using only RGB video—without relying on stereo cameras or LiDAR.

This approach mimics how lightweight AVs (e.g., delivery bots, two-wheelers) might operate in complex, unstructured road scenes common in developing countries.

---

##  Objectives

- Reconstruct **sparse local 3D maps** from monocular dashcam sequences
- Estimate **camera poses** and **depth** per frame
- Fuse geometric information to build consistent local maps
- Address challenges unique to **Indian driving environments**: occlusion, dense and heterogeneous traffic, lack of lane structure

---

##  Dataset

- **IDD-Multimodal**: A challenging dashcam dataset captured across diverse Indian cities
  - Urban and semi-urban settings
  - Varied lighting and weather conditions
  - Frame-wise RGB and auxiliary OBD metadata

---

##  Pipeline (In Progress)

### 1. Monocular Depth Estimation
- Using **Depth Anything V2** to predict dense depth maps from individual frames
- Outputs: per-frame depth images for sequences (`d0`, `d1`, etc.)

### 2. Pose Estimation & Sparse Reconstruction
- Using **COLMAP** with sequential matcher for monocular video-based pose estimation
- Intrinsics estimated internally using the `SIMPLE_RADIAL` model
- Outputs:
  - Sparse point cloud
  - Camera extrinsics and trajectory for each sequence

### 3. Depth-Geometry Fusion *(In Progress)*
- Plan to align predicted depth maps with COLMAP camera poses
- Fuse into a coherent local map via back-projection and visibility filtering

### 4. Scale Alignment *(Planned)*
- Use OBD speed readings to approximate real-world scale
- Optional: Temporal alignment for speed-aware trajectory estimation

---

##  Research Motivation

- **Monocular cameras** are inexpensive and already available in most vehicles
- **Lightweight mapping** approaches are essential for low-cost AV stacks
- **Indian road scenes** represent under-explored complexity in global datasets
- Potential to extend toward **semantic-aware 3D mapping** or **topological navigation**

---

## Next Steps

- Complete depth-pose fusion and generate per-sequence local maps
- Compare accuracy and sparsity across varying driving sequences
- Integrate semantic segmentation for map-level understanding *(future work)*
- Evaluate trajectory-scale alignment with OBD data

---

##  Applications

- Autonomous navigation for 2-wheelers and delivery vehicles
- Re-localization and loop closure in unstructured urban scenes
- Semantic simulation environments from real-world monocular video
- Low-cost robotics and edge-device mapping pipelines

---
##  Author

**Ananya Kulkarni**  
AI/ML Intern @ iHub-Data,IIIT-H 
[LinkedIn](https://linkedin.com/in/ananya-kulkarni-609213244) | [Google Scholar](https://scholar.google.com/citations?user=GddubbUAAAAJ) | [GitHub](https://github.com/ananya12k)

---

