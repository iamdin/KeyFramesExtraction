import os
import cv2
from typing import List
from flask import Flask, request, jsonify
from keyframes_extract_time import Time, handle_images_times

app = Flask(__name__)


@app.route('/submit', methods=['get'])
def add_task():
    start = int(request.args['start'])
    end = int(request.args['end'])
    print(start, end)
    store_keyframe('assessment.mp4', './assessment_frames',
                   handle_images_times('./frames_video'),
                   start * 1000,
                   end * 1000)
    return jsonify({'result': 'success'})


def store_keyframe(video_path: str, target_path: str, times: List[int], start: int, end: int):
    """
    根据标准视频关键帧的时间段，在评估视频中相应位置的前后两秒内，提取关键帧
    :return:
    """
    # 创建视频对象
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧数
    print(cap.get(cv2.CAP_PROP_FPS))

    print(frame_count)
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    no = 1
    pos = 0
    # 读取视频帧
    nex, frame = cap.read()
    print(start)
    print(end)
    print(times)
    while nex:
        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        if start <= int(milliseconds) <= end and pos < len(times) and \
                abs(int(milliseconds) - start - times[pos]) <= 2000:
            print(milliseconds)
            cv2.imwrite(f'{target_path}/{no}-{frame_count}_{Time(cap.get(cv2.CAP_PROP_POS_MSEC))}.jpg', frame)
            pos += 1

        if milliseconds > end:
            break
        no += 1
        nex, frame = cap.read()

    cap.release()


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="127.0.0.1", port=9999, debug=True)
# if __name__ == '__main__':
#     store_keyframe('assessment.mp4', './assessment_frames', handle_images_times('./frames_video'))
