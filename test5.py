import cv2
import numpy as np


def apply_geometric_transform(input_image, roll_angle, output_size):
    """
    对输入图片进行旋转和空间映射操作。

    参数：
    input_image: 输入图片，大小为 (m, n)
    roll_angle: 相机的 roll 角度（绕 Z 轴的旋转角度，单位为度数）
    output_size: 输出图片的大小，(p, q)

    返回：
    变换后的图片
    """
    input_height, input_width = input_image.shape[:2]
    output_height, output_width = output_size

    # 计算图片的中心
    center = (input_width // 2, input_height // 2)

    # 获取旋转矩阵 (基于 roll 角度)
    rotation_matrix = cv2.getRotationMatrix2D(center, roll_angle, 1.0)

    # 应用仿射变换进行旋转
    rotated_image = cv2.warpAffine(input_image, rotation_matrix, (input_width, input_height))

    # 使用 cv2.resize 进行自适应大小调整（如池化）
    resized_image = cv2.resize(rotated_image, (output_width, output_height))

    return resized_image


# 示例使用
input_image = np.random.rand(128, 256) * 255  # 输入大小为 128x256 的随机图像
input_image = input_image.astype(np.uint8)  # 转换为 8 位无符号整数类型

roll_angle = 30  # 相机的 roll 角度（度）
output_size = (90, 150)  # 目标输出大小为 90x150

# 应用几何变换
output_image = apply_geometric_transform(input_image, roll_angle, output_size)

# 展示结果
cv2.imshow("Transformed Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
