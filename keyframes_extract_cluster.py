import os
import cv2
import numpy as np
from typing import List
from matplotlib import pyplot as plt

threshold = float(0.9876543)


class FrameHist:
    """帧直方图"""

    def __init__(self, no: int, hist: List):
        self.no = no
        self.hist = hist


class FrameCluster:
    """帧聚类"""

    def __init__(self, cluster: List[FrameHist], center: FrameHist):
        self.cluster = cluster
        self.center = center

    def re_center(self):
        """重新计算聚类中心"""
        hist_sum = [0] * len(self.cluster[0].hist)
        for i in range(len(self.cluster[0].hist)):
            for j in range(len(self.cluster)):
                hist_sum[i] += self.cluster[j].hist[i]
        self.center.hist = [i / len(self.cluster) for i in hist_sum]


def similarity(frame1, frame2):
    """
    直方图计算法，两帧之间的相似度

    TODO
    此算法可优化：平均哈希算法，感知哈希算法，直方图计算法
    """
    s = np.vstack((frame1, frame2)).min(axis=0)
    similar = np.sum(s)
    # print(similar)
    return similar


def handle_video_frames(source_path: str, target_path: str):
    # 创建视频对象
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数
    height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 帧高，宽
    print(f'Frames Per Second: {fps}')
    print(f'Number of Frames : {frame_count}')
    print(f'Height of Video: {height}')
    print(f'Width of Video: {width}')

    times = 1
    frames, frame_hists = list(), list()
    while True:

        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR -> HSV 转换颜色空间
        # 统计颜色直方图，[h,s,v]:[0,1,2] 分为 [12,5,5]份
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 5, 5], [0, 256, 0, 256, 0, 256])
        # numpy 3维数组扁平化
        flatten_hists = hist.flatten()
        # 求均值
        flatten_hists /= height * width
        frame_hists.append(flatten_hists)

        # 显示3个通道的颜色直方图
        # plt.plot(flatten_hists[:12], label=f'H_{times}', color='blue')
        # plt.plot(flatten_hists[12:17], label=f'S_{times}', color='green')
        # plt.plot(flatten_hists[17:], label=f'V_{times}', color='red')
        # plt.legend(loc='best')
        # plt.xlim([0, 22])

        times += 1
    # plt.show()

    # 释放

    # 聚类,第一个自成一类
    clusters = list([FrameCluster([FrameHist(0, frame_hists[0])], FrameHist(0, frame_hists[0]))])

    for no, hist in enumerate(frame_hists[1:]):
        """
        将每一帧与已形成的聚类中心比较，取相似度最大值，其若小于阈值则自成一类，否则加入此类
        """
        max_ratio, clu_idx = 0, 0
        for i, clu in enumerate(clusters):
            sim_ratio = similarity(hist, clu.center.hist)
            if sim_ratio > max_ratio:
                max_ratio, clu_idx = sim_ratio, i

        # 最大相似度 与 阈值比较
        if max_ratio < threshold:
            """小于阈值，自成一类"""
            clusters.append(FrameCluster([FrameHist(no, hist)], FrameHist(no, hist)))
        else:
            clusters[clu_idx].cluster.append(FrameHist(no, hist))
            # 重新计算聚类中心
            clusters[clu_idx].re_center()
            clusters[clu_idx].center.no = no

    # 按帧序号排序
    clusters.sort(key=lambda x: x.center.no)

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # 保存
    for clu in clusters:
        cv2.imwrite(f'{target_path}/{clu.center.no}.jpg', frames[clu.center.no])

    cap.release()


if __name__ == '__main__':
    # 打开当前文件夹下的 video.mp4 视频
    source = os.path.join(os.path.abspath('.'), 'video.mp4')
    target = os.path.join(os.path.dirname(source), 'frames_video')
    handle_video_frames(source, target)
    # video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (464, 848))  # (1360,480)为视频大小
    # print(os.listdir(target))
    # for img_name in os.listdir(target):
    #     video_writer.write(cv2.imread(f'{target}/{img_name}'))
    # video_writer.release()
