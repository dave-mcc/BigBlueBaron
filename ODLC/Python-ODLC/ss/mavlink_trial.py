import mavsdk
import mavsdk.system
import mavsdk
import cv2
from ultralytics import YOLO

video = "ODLC/Python-ODLC/video/video.mp4"
yolo = YOLO('ODLC/Python-ODLC/best.pt')
currentframe = 0
#print(cv2.getBuildInformation())

choice  = int(input('choose either video(1) or camera(0)'))
while True:
    if choice ==1:
        cap = cv2.VideoCapture(video)
    elif choice == 0:
        cap = cv2.VideoCapture(0)

