import imageio
import numpy as np
import open3d as o3d
import torch
from torchvision.utils import save_image
from sklearn.neighbors import KNeighborsRegressor
from promptda.utils.depth_utils import visualize_depth
from PIL import Image

def load_ply(filename):
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)  # 点的坐标 (Nx3)
    colors = np.asarray(pcd.colors)  # 点的颜色 (Nx3)
    return points, colors

ply_path = './lidar_cam_front.ply'
points, _ = load_ply(ply_path)

h, w = [1080, 1920]
K = np.array([
    [1545, 0, 960],
    [0, 1545, 560],
    [0, 0, 1]
])

print("Depth range: ", points[:, 2].min(), points[:, 2].max())

pts_depth = np.zeros([1, h, w])
point_camera = points
uvz = point_camera[point_camera[:, 2] > 0]
uvz = uvz @ K.T
uvz[:, :2] /= uvz[:, 2:]
uvz = uvz[uvz[:, 1] >= 0]
uvz = uvz[uvz[:, 1] < h]
uvz = uvz[uvz[:, 0] >= 0]
uvz = uvz[uvz[:, 0] < w]
uv = uvz[:, :2]
uv = uv.astype(int)
# TODO: may need to consider overlap
pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
pts_depth = torch.from_numpy(pts_depth).float()

# KNN, K=4
# Find all points with depth values
valid_points = []
valid_depths = []
for i in range(h):
    for j in range(w):
        if pts_depth[0, i, j] > 0:
            valid_points.append([i, j])
            valid_depths.append(pts_depth[0, i, j])

valid_points = np.array(valid_points)
valid_depths = np.array(valid_depths)

if len(valid_points) > 0:
    # Create and train KNN model
    knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
    knn.fit(valid_points, valid_depths)
    
    # Find all points without depth values
    invalid_points = []
    for i in range(h):
        for j in range(w):
            if pts_depth[0, i, j] == 0:
                invalid_points.append([i, j])
    
    invalid_points = np.array(invalid_points)
    
    if len(invalid_points) > 0:
        # Use KNN to predict depth values
        pred_depths = knn.predict(invalid_points)
        
        # Fill in predicted depth values
        for idx, (i, j) in enumerate(invalid_points):
            pts_depth[0, i, j] = pred_depths[idx]

pts_depth = pts_depth.numpy()
depth_int = (pts_depth[0] * 1000).astype(np.uint16)
print(depth_int.shape)
print(depth_int.min(), depth_int.max())
Image.fromarray(depth_int).save('pts_depth.png')
print(pts_depth.shape)
print(pts_depth.min(), pts_depth.max())

depth_vis, depth_min, depth_max = visualize_depth(pts_depth[0, :, :], ret_minmax=True)
imageio.imwrite('pts_depth_vis.png', depth_vis)