"""
@File  :main.py
@Author:jyding
@Date  :2020/8/26 下午2:42
@Desc  :视频关键帧提取
"""

import os
import cv2
import time
import numpy as np
from typing import List


def keyframes_extract_interval(source_path: str, target_path: str, frames_frequency=25) -> None:
    """
    固定间隔提取视频关键帧，将关键帧保存在目标文件目录中
    :param source_path:视频源文件地址
    :param target_path:关键帧输出地址
    :param frames_frequency:固定帧数
    :return: None
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


def keyframes_extract_cluster(source_path: str, target_path: str, threshold: float = 0.973) -> None:
    """
    聚类实现视频关键帧提取，将关键帧保存在目标文件目录中
    :param source_path: 视频源文件地址
    :param target_path: 关键帧保存地址
    :param threshold: 图片相似度阈值 0 - 1， 默认为 0.973
    :return: None
    """

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

    class Frame:
        """帧直方图"""

        def __init__(self, no: int, hist: List):
            """帧序号、直方图"""
            self.no = no
            self.hist = hist

    class FrameCluster:
        """帧聚类"""

        def __init__(self, cluster: List[Frame], center: Frame):
            self.cluster = cluster
            self.center = center

        def re_center(self):
            """重新计算聚类中心"""
            hist_sum = [0] * len(self.cluster[0].hist)
            for i in range(len(self.cluster[0].hist)):
                for j in range(len(self.cluster)):
                    hist_sum[i] += self.cluster[j].hist[i]
            self.center.hist = [i / len(self.cluster) for i in hist_sum]

        def keyframe_no(self) -> int:
            """
            聚类中与聚类中心相似度最高的
            :return: 帧序号
            """
            no = self.cluster[0].no
            max_similar = 0
            for frame in self.cluster:
                similar = similarity(frame.hist, self.center.hist)
                if similar > max_similar:
                    max_similar, no = similar, frame.no
            return no

    def similarity(frame1, frame2):
        """
        直方图计算法，两帧之间的相似度
        :param frame1: 第一帧的直方图
        :param frame2: 第二帧的直方图
        :return: 相似度 0 - 1
        """
        s = np.vstack((frame1, frame2)).min(axis=0)
        similar = np.sum(s)
        return similar

    def handle_video_frames(video_path: str) -> List[Frame]:
        """
        处理视频获取所有帧的HSV直方图
        :param video_path: 视频路径
        :return: 帧对象数组
        """
        # 创建视频对象
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数
        height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 帧高，宽
        print(f'Frames Per Second: {fps}')
        print(f'Number of Frames : {frame_count}')
        print(f'Height of Video: {height}')
        print(f'Width of Video: {width}')

        no = 1
        frames = list()

        # 读取视频帧
        nex, frame = cap.read()
        while nex:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR -> HSV 转换颜色空间
            # 统计颜色直方图，[h,s,v]:[0,1,2] 分为 [12,5,5]份
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 5, 5], [0, 256, 0, 256, 0, 256])
            # numpy 3维数组扁平化
            flatten_hists = hist.flatten()
            # 求均值
            flatten_hists /= height * width
            frames.append(Frame(no, flatten_hists))

            # 显示3个通道的颜色直方图
            # plt.plot(flatten_hists[:12], label=f'H_{times}', color='blue')
            # plt.plot(flatten_hists[12:17], label=f'S_{times}', color='green')
            # plt.plot(flatten_hists[17:], label=f'V_{times}', color='red')
            # plt.legend(loc='best')
            # plt.xlim([0, 22])

            no += 1
            nex, frame = cap.read()

            # plt.show()

        # 释放
        cap.release()
        return frames

    def frames_cluster(frames: List[Frame]) -> List[FrameCluster]:
        """
        聚类
        :param frames: 帧对象数组
        :return: 聚类数组
        """
        # 第一个自成一类
        ret_clusters = [FrameCluster([frames[0]], Frame(0, frames[0].hist))]

        for frame in frames[1:]:
            """
            将每一帧与已形成的聚类中心比较，取相似度最大值，其若小于阈值则自成一类，否则加入此类
            """
            max_ratio, clu_idx = 0, 0
            for i, clu in enumerate(ret_clusters):
                sim_ratio = similarity(frame.hist, clu.center.hist)
                if sim_ratio > max_ratio:
                    max_ratio, clu_idx = sim_ratio, i

            # 最大相似度 与 阈值比较
            if max_ratio < threshold:
                """小于阈值，自成一类"""
                ret_clusters.append(FrameCluster([frame], Frame(0, frame.hist)))
            else:
                ret_clusters[clu_idx].cluster.append(frame)
                # 重新计算聚类中心
                ret_clusters[clu_idx].re_center()

        return ret_clusters

    def store_keyframe(video_path: str, target_path: str, frame_clusters: List[FrameCluster]) -> None:
        """
        从聚类中 保存视频关键帧
        :param video_path: 原视频地址
        :param target_path: 关键帧保存地址
        :param frame_clusters: 帧聚类数组
        :return: None
        """
        # 创建视频对象
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧数

        # video_path关键帧序号
        nos = set([cluster.keyframe_no() for cluster in frame_clusters])
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        no = 1
        # 读取视频帧
        nex, frame = cap.read()
        while nex:
            if no in nos:
                print(cap.get(cv2.CAP_PROP_POS_MSEC))
                cv2.imwrite(f'{target_path}/{no}-{frame_count}_{Time(cap.get(cv2.CAP_PROP_POS_MSEC))}.jpg', frame)
            no += 1
            nex, frame = cap.read()

    frames = handle_video_frames(source_path)
    clusters = frames_cluster(frames)
    store_keyframe(source_path, target_path, clusters)


def keyframes_extract_time(source_path: str, source_frames_path: str, target_path: str) -> None:
    """
    对应时间段提取关键帧，将关键帧保存在目标文件目录中
    :param source_path: 视频原视频地址
    :param source_frames_path: 标准视频关键帧文件夹地址，关键帧图片命名格式：关键帧序号-总帧数_XXh-XXm-XXs-XXms.jpg'
    :param target_path: 关键帧提取输出地址
    :return:
    """

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

    store_keyframe(source_path, target_path, handle_images_times(source_frames_path))


def keyframes_extract_three_frame_diff(source_path: str, target_path: str) -> None:
    """
    三帧差法提取视频关键帧，将关键帧保存在目标文件目录中
    :param source_path: 原视频文件地址
    :param target_path: 关键帧保存文件夹地址
    :return: None
    """
    cap = cv2.VideoCapture(source_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频文件输出参数设置
    one_frame = np.zeros((height, width), dtype=np.uint8)
    two_frame = np.zeros((height, width), dtype=np.uint8)
    three_frame = np.zeros((height, width), dtype=np.uint8)
    no = -1
    while cap.isOpened():
        no += 1
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
                cv2.imwrite(f'{target_path}/{no}.jpg', frame)

        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
    cap.release()


def keyframes_extract_real_time(target_path: str) -> None:
    """
    调用摄像头实时提取15秒关键帧,将关键帧保存在目标文件目录中
    :param target_path: 关键帧提取目标文件夹地址
    :return:
    """
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频文件输出参数设置
    out_fps = 12.0
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw1 = cv2.VideoWriter(f'{target_path}/frames_result.mp4', fourcc, out_fps, (width, height))
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
                # x, y, w, h = cv2.boundingRect(cnt)  # 找方框
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(f'{target_path}/{no}.jpg', frame)
                vw1.write(frame)

        cv2.imshow("frame", frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
    vw1.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
