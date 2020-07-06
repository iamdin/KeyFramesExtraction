import os
import cv2

source = os.path.join(os.path.abspath("."), 'video.mp4')
target = os.path.join(os.path.dirname(source), 'frames_video')

cap = cv2.VideoCapture(source)
print(cap.get(cv2.CAP_PROP_FPS))
