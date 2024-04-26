import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from classification.focalnet import FocalNet, focalnet_large_fl3
import os
from _meta import _IMAGENET_CATEGORIES, _IMAGENET_CATEGORIES2

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),  # 将图像调整为 256x256
    transforms.CenterCrop(224),  # 中心裁剪 224x224 大小的图像
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

# 加载数据集
data_dir = '../ImageNet21K-main/data/0.1/0.1output-10'
dataset = datasets.ImageFolder(data_dir, transform=preprocess)
# 创建数据加载器
batch_size = 250
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = focalnet_large_fl3()
model.eval()
total_correct = 0
total_images = 0
# 加载标签文件
label_dict = {}
with open('../ImageNet21K-main/label.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)  # 以第一个空格为分隔符，最多分割成两部分
        idx = parts[0]
        label = parts[1] if len(parts) > 1 else ''  # 如果标签不存在，则置为空字符串
        label_dict[idx] = label

label_map_file = "labelmap_22k_reorder.txt"
# 创建一个空字典来存储标签映射关系
label_map = {}
# 读取标签映射文件，并将其存储到字典中
with open(label_map_file, 'r') as file:
    for line in file:
        label, index = line.strip().split('\t')
        label_map[int(index)] = label
# # 将字典转换为数组
# labels = [None] * len(label_map)
# for index, label in label_map.items():
#     labels[index] = label
results_file = open('prediction_result.txt', 'a')  # 使用 'a' 模式以追加写入
# 保存准确率的文件
accuracy_file = open('accuracy.txt', 'a')  # 使用 'a' 模式以追加写入
# 对数据集进行预测
results_file.write(f"\n----------------------------------- {data_dir}------------------------------------\n")
for images, _ in data_loader:
    # images = images.to(device)  # 将图像移到 GPU 上
    with torch.no_grad():
        outputs = model(images)
    
    _, predicted = torch.max(outputs, 1)
    for i in range(len(images)):
        print(i)
        img_path = dataset.samples[total_images + i][0]  # 获取图片路径
        img_name = os.path.basename(img_path)  # 获取图片文件名
        true_label = label_dict[os.path.basename(os.path.dirname(img_path))]  # 获取真实标签
        # true_label = os.path.basename(os.path.dirname(img_path))  # 获取真实标签
        print(predicted[i])
        # predicted_label = label_map[int(predicted[i])]
        predicted_label =_IMAGENET_CATEGORIES[predicted[i]]
        results_file.write(f"图片名: {img_name}, 真实标签: {true_label}, 预测标签: {predicted_label}\n")
        if predicted_label == true_label:
        # if true_label in predicted_label:
            total_correct += 1
    total_images += len(images)  # 更新总样本数量
# 计算准确率
print(total_images)
accuracy = total_correct / total_images
accuracy_file.write(f"{data_dir}准确率: {accuracy:.2%}\n")
print("准确率:",accuracy)

# 关闭保存结果的文件
results_file.close()
accuracy_file.close()
