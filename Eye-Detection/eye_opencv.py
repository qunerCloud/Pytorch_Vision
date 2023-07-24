import torch,cv2,time
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #通常，建议将scaleFactor识别尺度设置为介于1.01到1.5之间的值。
    #通常，建议将minNeighbors重叠的候选区域设置为3或4。较大的minNeighbors值会导致合并更多的候选区域，从而减少重叠的检测结果，但可能会导致某些目标漏检。
    #较大的minSize值会减少误检，但可能会导致较小的目标无法被检测到。通常，建议根据实际应用场景选择合适的minSize值。
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=6, minSize=(50, 50))
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

cap = cv2.VideoCapture(0)
cap.open(0)
start_time = time.time()
frame_count = 0
fps = 0

while True:
    # 读取视频帧
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 在图像上绘制检测到的人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if not ret:
        break  
    frame = detect_eyes(frame)
    # 计算FPS
    frame_count += 1
    if frame_count >= 15:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        start_time = end_time
        frame_count = 0

    # 在窗口屏幕上写入“当前FPS”
    cv2.putText(frame, f'Current FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示处理后的图像
    cv2.imshow('Eye and Face Detection', frame)

    # 检测按键，如果按下 'q' 键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
