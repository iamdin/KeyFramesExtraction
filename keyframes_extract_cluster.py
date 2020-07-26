import os
import cv2
import numpy as np
from typing import List
from matplotlib import pyplot as plt

# TODO
# HSV颜空间下，图片相似度对比
# 多进程计算

# 相似度阈值
threshold = float(0.96)
# 打开当前文件夹下的 video.mp4 视频
source = os.path.join(os.path.abspath('.'), 'standard.mp4')
target = os.path.join(os.path.dirname(source), 'frames_video')


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

    TODO
    此算法可优化：平均哈希算法，感知哈希算法，直方图计算法
    """
    s = np.vstack((frame1, frame2)).min(axis=0)
    similar = np.sum(s)
    return similar


def handle_video_frames(video_path: str) -> List[Frame]:
    """
    处理视频
    :param video_path:
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
    :param frames:
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


def store_keyframe(video_path: str, target_path: str, frame_clusters: List[FrameCluster]):
    """
    从聚类中 保存视频关键帧
    :return:
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


if __name__ == '__main__':
    frames = handle_video_frames(source)
    clusters = frames_cluster(frames)
    store_keyframe(source, target, clusters)
