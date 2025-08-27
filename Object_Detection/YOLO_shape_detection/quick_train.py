import os
import torch
from ultralytics import YOLO
from load_data import prepare_yolo_dataset


def train_yolo_model(yaml_path="./data/dataset.yaml"):
    # 初始化模型
    model = YOLO("yolov8n.pt")
    # 训练参数
    train_args = {
        'data': yaml_path,           # 使用准备好的配置文件
        'epochs': 5,               # 训练轮数
        'imgsz': 640,              # 图像尺寸
        'batch': 8,                # 批次大小
        'device': 'cpu',           # 使用 CPU
        'lr0': 0.01,              # 学习率
        'patience': 10,            # 早停耐心值
        'save': True,              # 保存模型
        'project': './runs',       # 项目目录
        'name': 'quick_train',     # 实验名称
        'verbose': True            # 显示详细信息
    }
    try:
        # 开始训练
        print("开始训练 YOLO 模型...")
        results = model.train(**train_args)
        print(f"✅ 训练完成！模型保存在: {results.save_dir}")
        # 显示训练结果
        print(f"📊 训练结果:")
        print(f"   - 最佳模型: {results.save_dir}/weights/best.pt")
        print(f"   - 最新模型: {results.save_dir}/weights/last.pt")
        print("\n=== 总结 ===")
        print("✅ YOLO 格式数据集准备完成")
        print("✅ 模型训练完成")
        print("📁 训练结果: ./runs/quick_train/")
        print("📁 数据集: ./data/images/ 和 ./data/labels/")
        print("💡 可以使用训练好的模型进行推理了")       

        # ✅ 训练完成！模型保存在: runs\quick_train
        # 📊 训练结果:
        # - 最佳模型: runs\quick_train/weights/best.pt
        # - 最新模型: runs\quick_train/weights/last.pt

        # === 总结 ===
        # ✅ YOLO 格式数据集准备完成
        # ✅ 模型训练完成
        # 📁 训练结果: ./runs/quick_train/
        # 📁 数据集: ./data/images/ 和 ./data/labels/

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("请检查数据集配置和训练参数")

if __name__ == "__main__":
    train_yolo_model()

