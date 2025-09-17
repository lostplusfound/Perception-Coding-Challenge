import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Config
DATASET_DIR = "dataset"
BBOX_FILE = os.path.join(DATASET_DIR, "bbox_light.csv")
XYZ_DIR = os.path.join(DATASET_DIR, "xyz")
OUTPUT_PNG = "trajectory.png"

# Load bboxes
bboxes = pd.read_csv(BBOX_FILE)

# Average depth over patch centered on center of bounding box
def avg_bbox_depth(row, patch_size):
    frame_id, x1, y1, x2, y2 = row

    x_center = (x1 + x2)//2
    y_center = (y1 + y2)//2

    xyz = np.load(os.path.join(XYZ_DIR, f"depth{frame_id:06d}.npz"))["xyz"]

    patch = xyz[y_center - patch_size:y_center + patch_size, x_center - patch_size:x_center + patch_size, :3].reshape(-1, 3)
    mask = np.isfinite(patch).all(axis=1)
    valid = patch[mask]

    if len(valid) == 0:
        return np.array([np.nan, np.nan, np.nan])
        
    return valid.mean(axis=0)

# Compute depth over each row

positions_cam = []
for _, row in bboxes.iterrows():
    avg_depth = avg_bbox_depth(row, 3)

    if np.all(np.isfinite(avg_depth)):
        positions_cam.append(avg_depth)

positions_cam = np.array(positions_cam)  # shape (N,3)

# Camera -> World axes
# camera: X forward, Y right, Z up
# world:  X forward, Y left,  Z up  => flip Y
cam_world = positions_cam.copy()
cam_world[:, 1] = -cam_world[:, 1]   # Y_world = -Y_cam

# Rotate so that initial car–light line is aligned with +X
p0 = cam_world[0, :2]  # first light position wrt car (XY plane)
angle = -np.arctan2(p0[1], p0[0])  # rotation to +X
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])

light_world_rot = (R @ cam_world[:, :2].T).T

# Plot
plt.figure(figsize=(8,8))
plt.plot(light_world_rot[:,0], light_world_rot[:,1], "o-", color="blue", markersize=4, label="Trajectory")
plt.scatter(light_world_rot[0,0], light_world_rot[0,1], color="green", s=120, label="Start")
plt.scatter(light_world_rot[-1,0], light_world_rot[-1,1], color="red", s=120, label="End")
plt.scatter(0, 0, color="black", marker="*", s=140, label="Traffic light (origin)")

plt.xlabel("X (m)  -- forward from traffic light")
plt.ylabel("Y (m)  -- left from traffic light")
plt.title("Ego-Vehicle Trajectory (World Frame)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.savefig(OUTPUT_PNG, dpi=200)
plt.close()