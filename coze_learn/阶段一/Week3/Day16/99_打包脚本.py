"""
Day16材料打包脚本
将Day16目录下的所有文件打包为Day16.zip
"""

import os
import zipfile
import sys

def create_zip():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(os.path.dirname(current_dir), 'Day16.zip')
    
    # 要排除的文件列表
    exclude_files = ['打包脚本.py', 'Day16.zip']
    
    print(f"正在打包目录: {current_dir}")
    print(f"目标ZIP文件: {zip_path}")
    
    # 创建ZIP文件
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历目录中的所有文件
        for root, dirs, files in os.walk(current_dir):
            # 计算相对路径
            rel_root = os.path.relpath(root, current_dir)
            
            for file in files:
                # 排除特定文件
                if file in exclude_files:
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.join(rel_root, file) if rel_root != '.' else file
                
                # 添加到ZIP
                zipf.write(file_path, rel_path)
                print(f"  添加: {rel_path}")
    
    # 计算ZIP文件大小
    zip_size = os.path.getsize(zip_path)
    print(f"\n打包完成!")
    print(f"ZIP文件大小: {zip_size / 1024:.2f} KB")
    print(f"包含文件数: {len(zipf.namelist())}")
    
    return zip_path

if __name__ == '__main__':
    zip_file = create_zip()
    print(f"\nZIP文件路径: {zip_file}")