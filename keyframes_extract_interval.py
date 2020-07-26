import cv2
import os


def frames_extraction(source_path: str, target_path: str, frames_frequency=25) -> None:
    """
    video frames extraction with fixed frame interval

    :param source_path:
    :param target_path:
    :param frames_frequency:
    :return:
    """
    # output directory path
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # 创建视频对象
    cap = cv2.VideoCapture(source_path)
    print(f'Frame rate: {cap.get(cv2.CAP_PROP_FPS)}')
    print(f'Number of frames in the video file: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')

    cnt, times = 0, 0
    while True:

        # 读取视频帧
        ret, image = cap.read()

        if not ret:
            break

        # 指定帧率保存帧图片
        if times % frames_frequency == 0:
            cv2.imwrite(f'{target_path}/{cnt}_{times}.jpg', image)
            print(f'{cnt}_{times}.jpg')
            cnt += 1

        times += 1

    print('frames extraction finished')
    # 释放
    cap.release()


if __name__ == '__main__':
    # source_path = input("source video full path (only mp4):")
    # target_path = input("frames extraction save path:")
    # frames_frequency = int(input("frames frequency"))
    # 打开当前文件夹下的 video.mp4 视频
    source = os.path.join(os.path.abspath('.'), 'standard.mp4')
    target = os.path.join(os.path.dirname(source), 'frames_video')
    frames_extraction(source, target)
