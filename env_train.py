import time,os,torch,torchvision,numpy as np,torch.nn as nn,torch.nn.functional as F,matplotlib.pyplot as plt,torch.optim as optim,pandas as pd
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
plt.rcParams['axes.unicode_minus'] = False#显示负号
best_test_accuracy = 0 #提前定义变量
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                      ])
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                                     ])

#Path
dataset_dir = '/Volumes/qunerSSD/Model/鞠九兵' 
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')
print('训练集路径', train_path)
print('测试集路径', test_path)

#Load
train_dataset = datasets.ImageFolder(train_path, train_transform)
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)

#对应单射类别存入表
class_names = train_dataset.classes
n_class = len(class_names)
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
np.save('idx_to_labels.npy', idx_to_labels)
np.save('labels_to_idx.npy', train_dataset.class_to_idx)

#训练设置
EPOCHS = 100
BATCH_SIZE = 32
device = torch.device('mps') #注意这里的设备类型，Apple是mps,NVIDIA记得改cuda

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0
                          )
# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0
                         )

# 载入评估函数
def evaluate_testset(model, criterion, device, test_loader):
    loss_list = []
    labels_list = []
    preds_list = []
    model.eval()  # 设置模型为评估模式，不计算梯度
    with torch.no_grad():
        for images, labels in test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
    log_test = {}
    log_test['epoch'] = epoch
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return log_test

#训练日志-测试集 初始化ResNet-18模型 修改全连接层，使得全连接层的输出与当前数据集类别数对应 交叉熵损失函数 优化器
df_test_log = pd.DataFrame()
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, n_class)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#正式开始训练
for epoch in tqdm(range(EPOCHS)):
    model.train()  # 设置模型为训练模式，计算梯度
    for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # 前向预测，获得当前 batch 的预测结果
        loss = criterion(outputs, labels)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
        optimizer.zero_grad()
        loss.backward()  # 损失函数对神经网络权重反向传播求梯度
        optimizer.step()  # 优化更新神经网络权重
    log_test = evaluate_testset(model, criterion, device, test_loader)
    df_test_log = pd.concat([df_test_log, pd.DataFrame(log_test, index=[0])], ignore_index=True)
    print(log_test)
    # 保存最新的最佳模型文件
    if log_test['test_accuracy'] > best_test_accuracy:
        # 删除旧的最佳模型文件
        old_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        # 保存新的最佳模型文件
        best_test_accuracy = log_test['test_accuracy']
        new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(log_test['test_accuracy'])
        torch.save(model, new_best_checkpoint_path)
        print('保存新的最佳模型', new_best_checkpoint_path)

print('Full task finished.Save success!The Fucking Accuracy is '+str(best_test_accuracy))