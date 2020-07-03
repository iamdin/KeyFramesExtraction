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

    cap = cv2.VideoCapture(source_path)
    cnt, times = 0, 0
    while True:

        ret, image = cap.read()

        if not ret:
            break

        if times % frames_frequency == 0:
            cv2.imwrite(f'{target_path}/{cnt}_{times}.jpg', image)
            print(f'{cnt}_{times}.jpg')
            cnt += 1

        times += 1

    print('frames extraction finished')
    cap.release()


if __name__ == '__main__':
    # source_path = input("source video full path (only mp4):")
    # target_path = input("frames extraction save path:")
    # frames_frequency = int(input("frames frequency"))
    source = 'C:\\Users\\jinyang\\Desktop\\video.mp4'
    target = os.path.join(os.path.dirname(source), 'frames_video')
    frames_extraction(source, target)
