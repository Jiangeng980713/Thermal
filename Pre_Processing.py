import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os


def apply_affine_transform(input_image, roll_degrees, yaw_degrees, distance):
    """
    使用 OpenCV 对输入图像进行 roll 和 yaw 角度的仿射变换。

    参数：
    input_image: 输入图像，大小为 (h, w, c) 或 (h, w) 的灰度图像
    roll_degrees: 绕 Z 轴（roll）旋转的角度
    yaw_degrees: 绕 Y 轴（yaw）旋转的角度
    w: 图像的宽度，用于计算透视变换

    返回：
    经过仿射变换后的图像
    """
    h, w = input_image.shape[:2]
    center = (w // 2, h // 2)

    # 1. Roll 角度旋转 (平面内旋转)
    roll_matrix = cv2.getRotationMatrix2D(center, roll_degrees, 1)

    # 2. Yaw 角度（透视变换）
    yaw_radians = np.deg2rad(yaw_degrees)

    # 设定相机距离
    d = w * distance  # 透视距离

    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    pts2 = np.float32([[d * np.sin(yaw_radians), 0],
                       [w - d * np.sin(yaw_radians), 0],
                       [w, h],
                       [0, h]])

    yaw_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 进行 roll 旋转变换
    rotated_image = cv2.warpAffine(input_image, roll_matrix, (w, h))

    # 进行 yaw 变换
    transformed_image = cv2.warpPerspective(rotated_image, yaw_matrix, (w, h))

    return transformed_image


def adaptive_pooling(image, output_size):
    """
    使用 PyTorch 实现自适应池化，将图像缩放到目标大小 (p, q)。

    参数：
    image: 输入图像，大小为 (h, w, c) 或 (h, w)
    output_size: 目标大小 (p, q)

    返回：
    池化后的图像，大小为 (p, q)
    """

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

    # 使用自适应池化将图像缩放到指定大小
    pooled_image_tensor = F.adaptive_avg_pool2d(image_tensor, output_size)

    # 转换回 NumPy 数组
    pooled_image = pooled_image_tensor.squeeze(0).squeeze(0).numpy().astype(np.uint8)

    return pooled_image


def process_images(input_dir, output_dir, roll_angle, yaw_angle, output_size, distance):
    """
    批量处理输入目录中的所有图像，并将处理后的图像保存到输出目录。

    参数：
    input_dir: 输入图像目录
    output_dir: 输出图像目录
    roll_angle: Roll 角度
    yaw_angle: Yaw 角度
    output_size: 自适应池化后的目标大小 (p, q)
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # 读取图像
        input_image = cv2.imread(os.path.join(input_dir, image_file))

        # 进行仿射变换
        affine_transformed_image = apply_affine_transform(input_image, roll_angle, yaw_angle, distance)

        # 进行自适应池化
        output_image = adaptive_pooling(affine_transformed_image, output_size)

        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_dir, image_file), output_image)


# 示例使用
input_directory = 'path/to/input/images'  # 替换为输入图像目录路径
output_directory = 'path/to/output/images'  # 替换为输出图像目录路径
roll_angle = 30  # Roll 角度
yaw_angle = 15  # Yaw 角度
output_size = (90, 150)  # 自适应池化后的目标大小 (p, q)

# 计算相机到图像中间的位置与像素大小的比例
distance = 1

# 处理图像
process_images(input_directory, output_directory, roll_angle, yaw_angle, output_size, distance)

print("图像处理完成！")
