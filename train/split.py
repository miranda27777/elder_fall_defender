# -*- coding: utf-8 -*-
import os
from os import getcwd

# 设置参数
sets = ['train', 'val', 'test']
classes = ["fall", "sit", "stand", "lie"]  # 类别列表
base_dir = 'D:/StudyFiles/YOLOv8/fallDownPic/'


def process_dataset(image_set):
    """处理单个数据集划分（train/val/test）"""
    # 创建必要目录
    os.makedirs(os.path.join(base_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'dataSet_path'), exist_ok=True)

    # 读取图像ID列表
    imageset_path = os.path.join(base_dir, 'ImageSets', 'Main', f'{image_set}.txt')
    if not os.path.exists(imageset_path):
        print(f"警告：未找到 {image_set} 的划分文件 {imageset_path}")
        return

    with open(imageset_path) as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]

    # 写入数据集路径文件
    dataset_path_file = os.path.join(base_dir, 'dataSet_path', f'{image_set}.txt')
    with open(dataset_path_file, 'w') as list_file:
        for image_id in image_ids:
            # 写入图像路径
            img_path = os.path.join(base_dir, 'images', f'{image_id}.jpg')
            list_file.write(f"{img_path}\n")

            # 复制标注文件（如果存在）
            src_label = os.path.join(base_dir, 'Annotation', f'{image_id}.txt')
            dst_label = os.path.join(base_dir, 'labels', f'{image_id}.txt')

            if os.path.exists(src_label):
                with open(src_label, 'r') as src, open(dst_label, 'w') as dst:
                    dst.write(src.read())
            else:
                print(f"警告：未找到标注文件 {src_label}")


if __name__ == '__main__':
    print(f"当前工作目录: {getcwd()}")
    print(f"基础目录: {base_dir}")

    for image_set in sets:
        print(f"正在处理 {image_set} 数据集...")
        process_dataset(image_set)

    print("处理完成！")