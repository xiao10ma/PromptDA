import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from general_utils import save_ply
import shutil
def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q: 四元数 [w, x, y, z]
    return: 3x3旋转矩阵
    """
    # 如果q是列表，转换为numpy数组
    q = np.array(q)
    
    # 提取四元数分量
    w, x, y, z = q[0], q[1], q[2], q[3]

    # 计算旋转矩阵元素
    r11 = 1 - 2 * y * y - 2 * z * z
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y

    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x * x - 2 * z * z
    r23 = 2 * y * z - 2 * w * x

    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x * x - 2 * y * y

    # 创建旋转矩阵
    rotation_matrix = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])
    
    return rotation_matrix

# 初始化
nusc = NuScenes(version='v1.0-mini', dataroot='/SSD_DISK/datasets/nuscenes', verbose=True)

# 获取某一帧 sample
sample = nusc.sample[0]
lidar_token = sample['data']['LIDAR_TOP']

# 获取该帧的点云数据
lidar_data = nusc.get('sample_data', lidar_token)
lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

# 加载点云（原始坐标系是sensor坐标系）
pc = LidarPointCloud.from_file(lidar_path)  # shape = (4, N), 包括 x,y,z,intensity

points_lidar = pc.points[:3, :].T  # shape = (N, 3)

# 删除自车，mask
# 计算每个点到原点的距离
distances = np.linalg.norm(points_lidar, axis=1)

# 创建掩码，标记距离大于等于2.5m的点（即排除自车点）
mask = distances >= 2.5

# 应用掩码筛选点云
points_lidar = points_lidar[mask]

lidar_calib_token = lidar_data['calibrated_sensor_token']
lidar_calib = nusc.get('calibrated_sensor', lidar_calib_token)

lidar_calib_trans = np.eye(4)
lidar_calib_trans[:3, :3] = quaternion_to_rotation_matrix(lidar_calib['rotation'])
lidar_calib_trans[:3, 3] = np.array(lidar_calib['translation']) # lidar -> ego

points_ego = points_lidar @ lidar_calib_trans[:3, :3].T + lidar_calib_trans[:3, 3]

front_cam_token = sample['data']['CAM_FRONT']
front_cam_data = nusc.get('sample_data', front_cam_token)
front_cam_path = os.path.join(nusc.dataroot, front_cam_data['filename'])
# cp from front_cam_path to ./image.jpg
shutil.copy(front_cam_path, './image.jpg')

front_cam_calib_token = front_cam_data['calibrated_sensor_token']
front_cam_calib = nusc.get('calibrated_sensor', front_cam_calib_token)

front_cam_calib_trans = np.eye(4)
front_cam_calib_trans[:3, :3] = quaternion_to_rotation_matrix(front_cam_calib['rotation'])
front_cam_calib_trans[:3, 3] = np.array(front_cam_calib['translation']) # cam -> ego

ego2cam = np.linalg.inv(front_cam_calib_trans) # ego -> cam
points_cam = points_ego @ ego2cam[:3, :3].T + ego2cam[:3, 3]
save_ply(points_cam, 'lidar_cam_front.ply')

front_cam_token = sample['data']['CAM_FRONT_LEFT']
front_cam_data = nusc.get('sample_data', front_cam_token)
front_cam_path = os.path.join(nusc.dataroot, front_cam_data['filename'])

front_cam_calib_token = front_cam_data['calibrated_sensor_token']
front_cam_calib = nusc.get('calibrated_sensor', front_cam_calib_token)

front_cam_calib_trans = np.eye(4)
front_cam_calib_trans[:3, :3] = quaternion_to_rotation_matrix(front_cam_calib['rotation'])
front_cam_calib_trans[:3, 3] = np.array(front_cam_calib['translation']) # cam -> ego

ego2cam = np.linalg.inv(front_cam_calib_trans) # ego -> cam
points_cam = points_ego @ ego2cam[:3, :3].T + ego2cam[:3, 3]

save_ply(points_cam, 'lidar_cam_left.ply')










