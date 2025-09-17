# Perception Coding Challenge Writeup

### Introduction
This is a summary of the process for completing my trajectory coding challenge.
### Initial Thoughts
With very little experience in computer vision and zero experience with object detection and tracking, I was initially quite intimidated by this problem. I wasn't really sure how to approach it, so I asked ChatGPT for some help in understanding how I could solve the problem conceptually. 

I quickly realized that the mandatory parts of the problem, estimating trajectory using the traffic light as a reference, would be easier than I expected, as the dataset already includes the traffic light tracking data, and a depth map. With an increased confidence, I began laying out how I would approach this problem
### Approach
My initial plan consisted of four main steps:
1. For each frame, read in the bounding box for the traffic light from bbox_light.csv 
2. Determine the position of the center of the bounding box in camera coordinates, by reading in the corresponding `frame_xxxxxx.npz` and getting the (X, Y, Z) coordinates for the center point
3. Transform these (X, Y, Z) coordinates to be relative to the traffic light
4. Plot the coordinates 

These steps seemed mostly straightforward to me, except for the 3rd step, which I predicted would probably get pretty math intensive. 
### Implementation
First, I laid out some basic config code, that handled libraries, directories, and filenames:
```import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Config
DATASET_DIR = "dataset"
BBOX_FILE = os.path.join(DATASET_DIR, "bbox_light.csv")
XYZ_DIR = os.path.join(DATASET_DIR, "xyz")
OUTPUT_PNG = "trajectory.png"
```
Next, I used pandas to load bboxes as a dataframe: 
```# Load bboxes
bboxes = pd.read_csv(BBOX_FILE)
```
After loading in bboxes, I then handled steps 2 and 3 in a helper function:
```# Average depth over patch centered on center of bounding box
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
```
While at first I just used the center point, I found that using a small patch of points around the center increased stability a bit. This function first calculates the center point, then loads the depth map file corresponding to the frame. It then slices the depth map to only the small patch of points that are relevant. Sometimes, the values of the depth map are inf  or nan , so I had to use a mask to select only the rows from the slice that are finite. Finally, I calculate the mean of the three columns, X, Y, and Z, and return them. 

A major bug that I had was assuming that bbox_light.csv  was formatted as frame_id, x1, x2, y1, y2 , when in reality it is frame_id, x1, y1, x2, y2 . This bug lead to immense time wasted on debugging, as I would get inf  or nan  values for the majority of the frames, leading to a really weird path shape, and consisting of just a few points separated by large gaps. I'm definitely going to make sure to read more carefully in the future.

I call this helper method in a for loop, storing the depth values in an array, then converting it to a numpy array:
```# Compute depth over each row

positions_cam = []
for _, row in bboxes.iterrows():
    avg_depth = avg_bbox_depth(row, 3)

    if np.all(np.isfinite(avg_depth)):
        positions_cam.append(avg_depth)


positions_cam = np.array(positions_cam)  # shape (N,3)
```
Next is the fun part: transforming these coordinates to be relative to the traffic light instead of the camera, and ensuring the required world frame setup: 

- The origin is directly under the traffic light on the ground.
- The Z-axis passes upward through the traffic light.
- At t = 0, the line joining the car and the traffic light is aligned with the +X axis.
- This defines a right-handed coordinate system with (X forward, Y left, Z up).
For this step, I didn't really know where to start, so I asked ChatGPT for help understanding how to achieve this conceptually.
First, the camera's coordinate system has Y to the right, so we need to flip it to make the coordinate system right handed. 
```# Camera -> World axes
# camera: X forward, Y right, Z up
# world:  X forward, Y left,  Z up  => flip Y
cam_world = positions_cam.copy()
cam_world[:, 1] = -cam_world[:, 1]   # Y_world = -Y_cam
```
Then, we need to make sure that the +X axis is the line joining the car and traffic light when t = 0. Basically, we need to rotate all the coordinates. The angle we rotate by is the opposite of the angle between the line joining the car and traffic light when t = 0 and +X axis, or just the arctan of the initial y and x coordinates of the traffic light. 

I wasn't really sure how to achieve the rotation though, so I just had ChatGPT write an implementation for me, which uses a rotation matrix. 
```# Rotate so that initial car–light line is aligned with +X
p0 = cam_world[0, :2]  # first light position wrt car (XY plane)
angle = -np.arctan2(p0[1], p0[0])  # rotation to +X
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])

light_world_rot = (R @ cam_world[:, :2].T).T
```
With all the hard work done, we just have to plot our data. Frankly, I'm not that familiar with python as a whole, so all the data analysis and plotting functions are quite unfamiliar with me. Therefore, I just had ChatGPT write me some code to generate a plot based on the specifications, which works well enough: 
```# Plot
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
```

### Conclusion
This challenge was initially intimidating but ultimately incredibly rewarding. While my plotted trajectory doesn’t perfectly match the car’s path in the video, I’m still proud of the solution I developed, especially since I had to learn so many things on the fly.