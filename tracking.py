import cv2
import numpy as np
import mediapipe as mp
import time

#（1）视频捕获  # 0代表电脑自带的摄像头
cap = cv2.VideoCapture(0) 
 
#（2）手部检测
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mpDraw = mp.solutions.drawing_utils

pTime = 0                                                               #处理一张图像前的时间
cTime = 0                                                               #一张图处理完的时间

points = []                                                             # 存储关节点8的坐标
start_time = time.time()                                                # 记录开始时间

blank_image = np.ones((720, 1280, 3), np.uint8) * 255                   # 画布

# （3）处理视频图像
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:                                    
        for handLms in results.multi_hand_landmarks:
            for index, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                
                if index == 8:
                    # print(index, cx, cy)
                    points.append((cx, cy))
                    cv2.circle(img, (cx, cy), 12, (255,0,255), cv2.FILLED)
                    
                    if len(points) > 1:
                        for i in range(1, len(points)):
                            cv2.line(img, points[i - 1], points[i], (255, 255, 255), 2)
                            cv2.line(blank_image, points[i - 1], points[i], (255, 0, 0), 2)

    # 执行时间
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # 把fps显示在窗口上；img画板；取整的fps值；显示位置的坐标
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 10:  # 这里暂时是设置10秒后退出
        cv2.imwrite("hand_trajectory.jpg", blank_image)                     # 保存关节点8轨迹
        break

cap.release()
cv2.destroyAllWindows()