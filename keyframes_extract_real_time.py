import cv2

# 调取摄像头
cap = cv2.VideoCapture(0)
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 视频文件输出参数设置
out_fps = 12.0
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vw1 = cv2.VideoWriter('./v3.mp4', fourcc, out_fps, (width, height))
vw2 = cv2.VideoWriter('./v4.mp4', fourcc, out_fps, (width, height))

# 初始化当前帧的前两帧及帧差
last_frame1 = last_frame2 = None
frame_delta1 = frame_delta2 = None

no = 0
nex, frame = cap.read()

# 遍历视频的每一帧
while nex:

    # 调整该帧的大小
    frame = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    # 如果第一二帧是None，对其进行初始化,计算第一二帧的不同
    if last_frame1 is None:
        last_frame1 = frame
        continue
    if last_frame2 is None:
        last_frame2 = frame
        frame_delta1 = cv2.absdiff(last_frame1, last_frame2)  # 帧差一
        continue

    # 计算当前帧和前帧的不同,计算三帧差分
    frame_delta2 = cv2.absdiff(last_frame2, frame)  # 帧差二
    thresh1 = cv2.bitwise_and(frame_delta1, frame_delta2)  # 图像与运算
    thresh2 = thresh1

    # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
    last_frame1, last_frame2 = last_frame2, frame
    frame_delta1 = frame_delta2

    # 结果转为灰度图
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)

    # 图像二值化
    thresh1 = cv2.threshold(thresh1, 25, 255, cv2.THRESH_BINARY)[1]

    # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
    thresh1 = cv2.dilate(thresh1, None, iterations=3)
    thresh1 = cv2.erode(thresh1, None, iterations=1)

    # 阀值图像上的轮廓位置
    cts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    for c in cts:
        # 忽略小轮廓，排除误差
        if cv2.contourArea(c) < 1000:
            continue

        # 计算轮廓的边界框，在当前帧中画出该框
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(f'./extract_result/{no}.jpg', frame)

    # 显示当前帧
    cv2.imshow("frame", frame)
    cv2.imshow("thresh1", thresh1)
    cv2.imshow("thresh2", thresh2)
    # 保存视频
    vw1.write(frame)
    vw2.write(thresh2)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    no += 1
    nex, frame = cap.read()

# 清理资源并关闭打开的窗口
vw1.release()
vw2.release()
cap.release()
cv2.destroyAllWindows()
