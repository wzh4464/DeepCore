import os

def rename_folders(start_path):
    """
    递归搜索并重命名文件夹
    :param start_path: 起始搜索路径
    :return: 被重命名的文件夹数量
    """
    count = 0
    
    # 确保起始路径存在
    if not os.path.exists(start_path):
        print(f"路径不存在: {start_path}")
        return count
    
    try:
        # 遍历所有文件和文件夹
        for root, dirs, files in os.walk(start_path, topdown=False):
            # 先处理文件夹名称
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                if dir_name == "liveval":
                    new_path = os.path.join(root, "liveval")
                    try:
                        os.rename(full_path, new_path)
                        print(f"已重命名: {full_path} -> {new_path}")
                        count += 1
                    except OSError as e:
                        print(f"重命名失败 {full_path}: {e}")
                        
    except Exception as e:
        print(f"遍历目录时出错: {e}")
    
    return count

def main():
    # 使用当前目录作为起始点
    current_dir = os.getcwd()
    print(f"开始搜索目录: {current_dir}")
    
    # 执行重命名操作
    renamed_count = rename_folders(current_dir)
    
    # 输出结果
    if renamed_count > 0:
        print(f"\n总共重命名了 {renamed_count} 个文件夹")
    else:
        print("\n没有找到需要重命名的文件夹")

if __name__ == "__main__":
    main()