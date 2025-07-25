# Replace LineSet with PointCloud of points with color
import open3d as o3d
import numpy as np
import pandas as pd

df = pd.read_csv("obd.csv")

# --- Parsing time manually ---
def parse_timestamp_custom(ts):
    try:
        parts = ts.split("-")
        minutes = int(parts[0])
        seconds = int(parts[1])
        millis = int(parts[2])
        micros = int(parts[3])
        return minutes * 60 + seconds + (millis * 1e-3) + (micros * 1e-6)
    except:
        return np.nan

df["time_sec"] = df["timestamp"].apply(parse_timestamp_custom)
df = df.dropna()
df["time_sec"] -= df["time_sec"].iloc[0]

# --- Integrate speed to get position ---
dt = np.gradient(df["time_sec"].values)
x = np.cumsum(df["speed"].values * dt)
y = np.zeros_like(x)
z = np.zeros_like(x)
points = np.vstack([x, y, z]).T

# Color all points red
colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(points), 1))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save to PLY
o3d.io.write_point_cloud("obd_trajectory_pointcloud.ply", pcd)
