import os
import cv2
import numpy as np
from typing import List


class Time:
    """帧时间"""

    def __init__(self, milliseconds: float):
        self.second, self.millisecond = divmod(milliseconds, 1000)
        self.minute, self.second = divmod(self.second, 60)
        self.hour, self.minute = divmod(self.minute, 60)

    def __str__(self):
        return f'{str(int(self.hour)) + "h-" if self.hour else ""}' \
               f'{str(int(self.minute)) + "m-" if self.minute else ""}' \
               f'{str(int(self.second)) + "s-" if self.second else ""}' \
               f'{str(int(self.millisecond)) + "ms"}'


def name2milliseconds(name: str) -> int:
    time = name.split('_')[-1].split('.')[0]

    def strip_alpha(s: str) -> str:
        return "".join([i for i in s if '0' <= i <= '9'])

    times = [int(strip_alpha(s)) for s in time.split('-')]
    milliseconds = times[-1]
    milliseconds += times[-2] * 1000 if len(times) > 1 else 0
    milliseconds += times[-3] * 60 * 1000 if len(times) > 2 else 0
    milliseconds += times[-4] * 60 * 60 * 1000 if len(times) > 3 else 0
    return milliseconds


def handle_images_times(dir_path: str) -> List[int]:
    names = os.listdir(dir_path)
    return sorted([name2milliseconds(name) for name in names])


def store_keyframe(video_path: str, target_path: str, times: List[int]):
    """
    根据标准视频关键帧的时间段，在评估视频中相应位置的前后两秒内，提取关键帧
    :return:
    """
    # 创建视频对象
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧数

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    no = 1
    pos = 0
    # 读取视频帧
    nex, frame = cap.read()
    while nex:
        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos < len(times) and abs(int(milliseconds) - times[pos]) <= 500:
            cv2.imwrite(f'{target_path}/{no}-{frame_count}_{Time(cap.get(cv2.CAP_PROP_POS_MSEC))}.jpg', frame)
            pos += 1
        no += 1
        nex, frame = cap.read()

    cap.release()


if __name__ == '__main__':
    store_keyframe('assessment.mp4', './assessment_frames', handle_images_times('./frames_video'))
