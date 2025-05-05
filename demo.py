from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth
import numpy as np
from general_utils import save_ply

def scale_intrinsics(K, orig_size, new_size):
    """
    缩放内参矩阵 K 以适配新的图像尺寸。
    orig_size: (H, W) 原始尺寸
    new_size: (H, W) 新尺寸
    """
    scale_y = new_size[0] / orig_size[0]
    scale_x = new_size[1] / orig_size[1]

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled

DEVICE = 'cuda'
image_path = "./image.jpg"
prompt_depth_path = "./pts_depth.png"
image = load_image(image_path).to(DEVICE)
prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters

model = PromptDA.from_pretrained("./model.ckpt").to(DEVICE).eval()
depth = model.predict(image, prompt_depth) # HxW, depth in meters

save_depth(depth, prompt_depth=prompt_depth, image=image)

# unproject depth to points
K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984], 
    [0.0, 1266.417203046554, 491.50706579294757], 
    [0.0, 0.0, 1.0]
])

K = scale_intrinsics(K, orig_size=(900, 1600), new_size=(560, 1008))

# 将深度从 PyTorch 张量转换为 NumPy 数组
depth_np = depth.detach().cpu().numpy()

# 生成图像平面上的坐标网格
height, width = depth_np.shape[-2:]
u, v = np.meshgrid(np.arange(width), np.arange(height))
u = u.flatten()
v = v.flatten()
z = depth_np.flatten()

# 忽略零深度或无效深度的点
valid_mask = z > 0
u = u[valid_mask]
v = v[valid_mask]
z = z[valid_mask]

# 将像素坐标转换为归一化相机坐标
x = (u - K[0, 2]) / K[0, 0]
y = (v - K[1, 2]) / K[1, 1]

# 计算3D点云坐标
points = np.zeros((len(z), 3))
points[:, 0] = x * z  # X坐标
points[:, 1] = y * z  # Y坐标
points[:, 2] = z      # Z坐标（深度）

# 将相机坐标系的点云转换为常见的可视化坐标系（可选，取决于可视化工具）
# 常见的变换是：z轴向前，y轴向下，x轴向右
# 这里假设原坐标系是：z轴向前，y轴向上，x轴向右
# points[:, 1] = -points[:, 1]  # 如果需要翻转y轴

# 保存点云
save_ply(points, 'points.ply')