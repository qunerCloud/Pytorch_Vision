import os,cv2,torch,time,numpy as np,pandas as pd,torch.nn.functional as F,matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from torchvision import models

font = ImageFont.truetype("SimHei.ttf", size=60)
device = torch.device('mps')
print('Using Fucking device:', device)
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
model = torch.load('/Users/quner/AI/checkpoint/best-1.000.pth', map_location=torch.device('mps'))
model = model.eval().to(device)
from torchvision import transforms
# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 处理帧函数
def process_frame(img):
    # 记录该帧开始处理的时间
    start_time = time.time()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 PIL
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
    top_n = torch.topk(pred_softmax, 2)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析置信度
    # 使用PIL绘制中文
    draw = ImageDraw.Draw(img_pil)
    for i in range(len(confs)):
        pred_class = idx_to_labels[pred_ids[i]]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
        # 文字坐标，中文字符串，字体，bgra颜色
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
    img = np.array(img_pil)  # PIL 转 array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB转BGR
    # 记录该帧处理完毕的时间
    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4, cv2.LINE_AA)
    return img

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)
# 打开cap
cap.open(0)
# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error,Cant get img.')
        break
    # 处理帧函数
    frame = process_frame(frame)
    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)
    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break
# 关闭摄像头
cap.release()
cv2.destroyAllWindows()