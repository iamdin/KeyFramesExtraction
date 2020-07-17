import cv2
import os
from keyframes_extract_cluster import handle_video_frames, similarity


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
    standard = handle_video_frames(os.path.join(os.path.abspath('.'), 'full_video.mp4'))
    assessment = handle_video_frames(os.path.join(os.path.abspath('.'), 'assessment.mp4'))
    max_similarity = float('-inf')
    for i, frame1 in enumerate(standard):
        similar = max([similarity(frame1.hist, frame2.hist) for frame2 in assessment])
        max_similarity = max(max_similarity, similar)
        print(f'standard frame-{i} max : {similar}')
    print(f'max : {max_similarity}')
