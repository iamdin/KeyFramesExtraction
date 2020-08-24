import cv2
import time
import numpy as np


def three_frame_differencing(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频文件输出参数设置
    out_fps = 12.0
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw1 = cv2.VideoWriter('./v3.mp4', fourcc, out_fps, (width, height))
    vw2 = cv2.VideoWriter('./v4.mp4', fourcc, out_fps, (width, height))
    one_frame = np.zeros((height, width), dtype=np.uint8)
    two_frame = np.zeros((height, width), dtype=np.uint8)
    three_frame = np.zeros((height, width), dtype=np.uint8)
    no = -1
    start = time.time()
    while cap.isOpened() and time.time() - start < 15:
        no += 1
        print(no)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(src=frame, dsize=(width, height),
                           interpolation=cv2.INTER_CUBIC)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray
        abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
        # 二值，大于40的为255，小于0
        _, thresh1 = cv2.threshold(abs1, 25, 255, cv2.THRESH_BINARY)

        abs2 = cv2.absdiff(two_frame, three_frame)
        _, thresh2 = cv2.threshold(abs2, 25, 255, cv2.THRESH_BINARY)

        binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erode = cv2.erode(binary, kernel)  # 腐蚀
        dilate = cv2.dilate(erode, kernel)  # 膨胀
        dilate = cv2.dilate(dilate, kernel)  # 膨胀

        cnts, _ = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for cnt in cnts:
            if 1000 < cv2.contourArea(cnt) < 40000:
                x, y, w, h = cv2.boundingRect(cnt)  # 找方框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(f'./extract_result/{no}.jpg', frame)
                cv2.imwrite(f'./extract_result_binary/{no}.jpg', binary)
                vw1.write(frame)
                vw2.write(binary)

        cv2.imshow("binary", binary)
        cv2.imshow("dilate", dilate)
        cv2.imshow("frame", frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
    vw1.release()
    vw2.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    three_frame_differencing(0)
