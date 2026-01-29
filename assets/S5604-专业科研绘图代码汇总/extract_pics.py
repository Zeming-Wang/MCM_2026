import os
import shutil
import json
import base64
from pathlib import Path


def is_image_file(filename):
    """判断文件是否为图片文件"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif']
    return any(filename.lower().endswith(ext) for ext in image_extensions)


def extract_images_from_ipynb(ipynb_path, output_dir):
    """从.ipynb文件中提取图片并保存"""
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    base_filename = Path(ipynb_path).stem  # 获取不含扩展名的文件名
    image_index = 1

    for cell in notebook['cells']:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output and 'image/png' in output['data']:
                    image_data = output['data']['image/png']
                    image_bytes = base64.b64decode(image_data)
                    image_filename = f"{base_filename}_{image_index}.png"
                    image_path = Path(output_dir) / image_filename

                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_bytes)

                    image_index += 1


def copy_image_files(src_dir, dst_dir, exclude_dirs):
    """复制图片文件到目标目录, 排除某个目录"""
    for root, dirs, files in os.walk(src_dir):
        # 排除某个目录
        for exclude_dir in exclude_dirs:
            if exclude_dir in dirs:
                dirs.remove(exclude_dir)
        for file in files:
            if is_image_file(file):
                src_file_path = Path(root) / file
                dst_file_path = Path(dst_dir) / file
                dst_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file_path, dst_file_path)


def process_directory(src_dir, dst_dir, exclude_dirs):
    """处理指定目录，复制图片文件，从.ipynb中提取图片并保存"""
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    copy_image_files(src_dir, dst_dir, exclude_dirs)

    for root, dirs, files in os.walk(src_dir):
        # 排除某个目录
        for exclude_dir in exclude_dirs:
            if exclude_dir in dirs:
                dirs.remove(exclude_dir)
        for file in files:
            if file.endswith('.ipynb'):
                ipynb_path = Path(root) / file
                extract_images_from_ipynb(ipynb_path, dst_dir)


if __name__ == "__main__":
    source_directory = input("请输入要遍历的源目录路径：")
    destination_path = input("请输入图片保存的目标路径：")
    exclude_dirs = ['.idea', '.git', '.ipynb_checkpoints', '图片汇总', '__pycache__']
    process_directory(source_directory, destination_path, exclude_dirs)