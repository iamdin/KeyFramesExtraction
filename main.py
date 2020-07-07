import cv2
import os


def handle_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    print(f'Frame rate: {cap.get(cv2.CAP_PROP_FPS)}')
    print(f'Number of frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
    print(f'width of video: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
    print(f'height of video: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')

    cnt, times = 0, 0
    while True:

        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        times += 1

    print('frames extraction finished')
    # 释放
    cap.release()


if __name__ == '__main__':
    source = os.path.join(os.path.abspath('.'), 'video.mp4')
    target = os.path.join(os.path.dirname(source), 'frames_video')
    handle_video(source)
