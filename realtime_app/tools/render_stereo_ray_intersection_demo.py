"""
绘制双目立体视觉系统中射线相交的示意图

展示左右相机如何通过射线三角化确定关节点的三维位置。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import json


class Arrow3D(FancyArrowPatch):
    """3D箭头"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def create_skeleton_pose():
    """创建一个标准的竖直人体骨架（简化版，聚焦下肢）"""
    # 定义关节点位置（单位：mm，原点在髋部中心）
    joints = {
        # 髋部
        'left_hip': np.array([-100, 0, 0]),
        'right_hip': np.array([100, 0, 0]),

        # 膝盖
        'left_knee': np.array([-100, 0, -400]),
        'right_knee': np.array([100, 0, -400]),

        # 踝关节
        'left_ankle': np.array([-100, 0, -800]),
        'right_ankle': np.array([100, 0, -800]),

        # 肩膀（用于显示上半身）
        'left_shoulder': np.array([-150, 0, 400]),
        'right_shoulder': np.array([150, 0, 400]),
        'neck': np.array([0, 0, 450]),
    }

    # 定义骨架连接
    connections = [
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle'),
        ('left_hip', 'left_shoulder'),
        ('right_hip', 'right_shoulder'),
        ('left_shoulder', 'neck'),
        ('right_shoulder', 'neck'),
        ('left_shoulder', 'right_shoulder'),
    ]

    return joints, connections


def create_camera_geometry():
    """创建相机几何（基于实际标定数据）"""
    # 相机中心位置（mm）
    left_camera = np.array([-185, 342, -165])
    right_camera = np.array([243, 302, -161])

    # 相机光轴方向（大致朝向人体）
    left_optical_axis = np.array([185, -342, 165]) / np.linalg.norm([185, -342, 165])
    right_optical_axis = np.array([-243, -302, 161]) / np.linalg.norm([-243, -302, 161])

    return {
        'left_center': left_camera,
        'right_center': right_camera,
        'left_axis': left_optical_axis,
        'right_axis': right_optical_axis
    }


def plot_camera_cone(ax, center, direction, size=50, color='gray', alpha=0.3):
    """绘制相机锥体"""
    # 创建锥体的基座
    cone_length = size * 1.5
    cone_radius = size * 0.5

    # 计算锥体的末端点
    end_point = center + direction * cone_length

    # 创建锥体基座的圆
    theta = np.linspace(0, 2*np.pi, 20)

    # 计算垂直于光轴的两个正交向量
    if abs(direction[2]) < 0.9:
        v1 = np.cross(direction, np.array([0, 0, 1]))
    else:
        v1 = np.cross(direction, np.array([0, 1, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(direction, v1)
    v2 = v2 / np.linalg.norm(v2)

    # 基座圆上的点
    circle_x = center[0] + cone_radius * (np.cos(theta) * v1[0] + np.sin(theta) * v2[0])
    circle_y = center[1] + cone_radius * (np.cos(theta) * v1[1] + np.sin(theta) * v2[1])
    circle_z = center[2] + cone_radius * (np.cos(theta) * v1[2] + np.sin(theta) * v2[2])

    # 绘制从中心到基座圆的线（形成锥体）
    for i in range(0, len(theta), 2):
        ax.plot([center[0], circle_x[i]],
                [center[1], circle_y[i]],
                [center[2], circle_z[i]],
                color=color, alpha=alpha, linewidth=0.5)

    # 绘制基座圆
    ax.plot(circle_x, circle_y, circle_z, color=color, alpha=alpha, linewidth=1)


def plot_ray_intersection_scene(joint_name, output_path, view_elev=None, view_azim=None):
    """
    绘制特定关节点的射线相交场景

    Args:
        joint_name: 关节点名称
        output_path: 输出图片路径
        view_elev: 视角仰角（度）
        view_azim: 视角方位角（度）
    """
    # 创建骨架和相机
    joints, connections = create_skeleton_pose()
    cameras = create_camera_geometry()

    # 目标关节点
    target_joint = joints[joint_name]

    # 创建图形
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制骨架（较细的线）
    for conn in connections:
        p1 = joints[conn[0]]
        p2 = joints[conn[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'b-', linewidth=1, alpha=0.6)

    # 绘制关节点（蓝色，较小）
    for name, pos in joints.items():
        if name == joint_name:
            # 目标关节点用红色
            ax.scatter(pos[0], pos[1], pos[2], c='red', s=50, alpha=1.0,
                      edgecolors='darkred', linewidths=1.5, zorder=100)
        else:
            ax.scatter(pos[0], pos[1], pos[2], c='blue', s=15, alpha=0.5, zorder=50)

    # 绘制相机
    plot_camera_cone(ax, cameras['left_center'], cameras['left_axis'],
                     size=40, color='darkblue', alpha=0.4)
    plot_camera_cone(ax, cameras['right_center'], cameras['right_axis'],
                     size=40, color='darkgreen', alpha=0.4)

    # 标记相机中心（较小的点）
    ax.scatter(*cameras['left_center'], c='darkblue', s=30, marker='o',
              edgecolors='black', linewidths=1, zorder=80)
    ax.scatter(*cameras['right_center'], c='darkgreen', s=30, marker='o',
              edgecolors='black', linewidths=1, zorder=80)

    # 绘制射线（从相机中心到关节点，较细的线）
    # 左相机射线
    left_ray_direction = target_joint - cameras['left_center']
    left_ray_end = cameras['left_center'] + left_ray_direction * 1.2  # 延伸一点
    ax.plot([cameras['left_center'][0], left_ray_end[0]],
            [cameras['left_center'][1], left_ray_end[1]],
            [cameras['left_center'][2], left_ray_end[2]],
            'darkblue', linewidth=1.5, alpha=0.8, linestyle='--', zorder=90)

    # 右相机射线
    right_ray_direction = target_joint - cameras['right_center']
    right_ray_end = cameras['right_center'] + right_ray_direction * 1.2
    ax.plot([cameras['right_center'][0], right_ray_end[0]],
            [cameras['right_center'][1], right_ray_end[1]],
            [cameras['right_center'][2], right_ray_end[2]],
            'darkgreen', linewidth=1.5, alpha=0.8, linestyle='--', zorder=90)

    # 设置坐标轴
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Z (mm)', fontsize=10)

    # 设置视角
    if view_elev is not None and view_azim is not None:
        ax.view_init(elev=view_elev, azim=view_azim)

    # 设置合适的显示范围（放大到关节点附近）
    zoom_radius = 300  # 显示半径
    ax.set_xlim(target_joint[0] - zoom_radius, target_joint[0] + zoom_radius)
    ax.set_ylim(target_joint[1] - zoom_radius, target_joint[1] + zoom_radius)
    ax.set_zlim(target_joint[2] - zoom_radius, target_joint[2] + zoom_radius)

    # 设置相等的坐标轴比例
    ax.set_box_aspect([1, 1, 1])

    # 移除网格使图片更清晰
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_path}")


def main():
    """生成三个不同关节点的射线相交图"""

    output_dir = r"D:\my_works\walker_pose_system\research_records\engineering_validation\stereo_ray_intersection_demo"

    import os
    os.makedirs(output_dir, exist_ok=True)

    # 定义三个关节点和它们的优化视角
    scenes = [
        {
            'joint': 'right_hip',
            'view_elev': -24,
            'view_azim': -160,
            'description': '右髋关节'
        },
        {
            'joint': 'right_knee',
            'view_elev': -16,
            'view_azim': 12,
            'description': '右膝关节'
        },
        {
            'joint': 'right_ankle',
            'view_elev': -4,
            'view_azim': 16,
            'description': '右踝关节'
        }
    ]

    # 生成每个场景
    metadata = {
        'description': '双目立体视觉射线三角化示意图',
        'coordinate_frame': '原点在髋部中心，Z轴竖直向上',
        'camera_configuration': '双鱼眼相机，倾斜约81度',
        'scenes': []
    }

    for scene in scenes:
        output_path = os.path.join(output_dir, f"{scene['joint']}_ray_intersection.png")

        plot_ray_intersection_scene(
            joint_name=scene['joint'],
            output_path=output_path,
            view_elev=scene['view_elev'],
            view_azim=scene['view_azim']
        )

        metadata['scenes'].append({
            'joint': scene['joint'],
            'description': scene['description'],
            'view_elevation_deg': scene['view_elev'],
            'view_azimuth_deg': scene['view_azim'],
            'output': output_path
        })

    # 保存元数据
    metadata_path = os.path.join(output_dir, 'scene_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n全部完成！输出目录: {output_dir}")
    print(f"元数据: {metadata_path}")


if __name__ == '__main__':
    main()
