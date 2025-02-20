import cv2
from ultralytics import YOLO
from pymavlink import mavutil
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import glob
import tensorflow as tf
import os

# Function to get class colors

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def drone_connect():
    the_connection = mavutil.mavlink_connection('udpin:localhost:14551')
    the_connection.wait_heartbeat()
    print('heartbeat from system(system %u component %u)'% (the_connection.target_system, the_connection.target_component))


##
## need to redefine the center coords upon testing drones camera
##
def object_detection():
    yolo = YOLO('best.pt')
    vid_option = input('OBJECT DETECTION: pick 0 to test with laptop camera, pick 1 to test with video.mp4, pick 2 to test with Test')
    if vid_option == '1':
        videoCap = cv2.VideoCapture('video/video.mp4')
        print("starting object detection with video/video.mp4")
    elif vid_option == '2':
        videoCap = cv2.VideoCapture('video/test.mp4')
        print("starting object detection with video/test.mp4")
    elif vid_option == '0':
        videoCap = cv2.VideoCapture(0)
        print("starting object detection with laptop camera")
    else:
        videoCap = cv2.VideoCapture('video/video.mp4')
        print('invalid option starting object detection on the defaut video/video.mp4')

    try:
        position_testing_dir = './pos_testing'
        if not os.path.exists(position_testing_dir):
            os.makedirs(position_testing_dir)
    except OSError:
        print('unable to open dir for testing position logic')
    current_frame = 0
    while True:
        ret, frame = videoCap.read()
        if not ret:
            continue
        results = yolo.track(frame, stream=True)
        for result in results:
            # get the classes names
            classes_names = result.names
            # iterate over each box
            for box in result.boxes:
                # check if confidence is greater than 40 percent
                if box.conf[0] > 0.4:
                    # get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    # convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # get the class
                    cls = int(box.cls[0])

                    # get the class name
                    class_name = classes_names[cls]

                    # get the respective colour
                    colour = getColours(cls)

                    # draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                 # put the class name and confidence on the image
                    cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                if box.conf > 0.8 and current_frame % 50 == 0:
                    image_name = str(position_testing_dir) + '/80_conf_frame_'+str(current_frame)+'.jpg'
                    [x_1, y_1, w_1, h_1] = box.xywh[0]
                    x_1,y_1,w_1,h_1 = int(x_1),int(y_1),int(w_1),int(h_1)
                    '''
                        center_position_x and center_position_y may need to be determined using np.reshape(w, h) or similar function to ensure consistency
                    '''
                    
                    H,W = frame.shape[:2]
                    center_position_x = int(W/2)
                    center_position_y = int(H/2)
                    get_position_relative(center_position_x,center_position_y, x_1, y_1,current_frame)
                    new_image = cv2.line(frame,(int(center_position_x),int(center_position_y)),(x_1,y_1),(255,0,0),5)
                    cv2.imwrite(image_name, new_image)
        current_frame = current_frame+1   
        # show the image
        cv2.imshow("frame", frame)
        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCap.release()
    cv2.destroyAllWindows()

height = 55
image_real_x=70
image_real_y=70
x_factor = image_real_x/640
y_factor = image_real_y/640
def get_position_relative(center_position_x,center_position_y,box_x, box_y, frame_num):
    directory_name = './plots_for_testing'
    try:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    except IOError:
        print(f'error making or finding the directory {directory_name}')
    plot_save_name = str(directory_name)+'/plot'+str(frame_num)
    
    x_unit = 1
    y_unit = 1
    xdist = box_x - center_position_x 
    ydist = box_y - center_position_y
    
    angle = np.arccos((ydist- y_unit)**2/(xdist - x_unit)**2)
    d_vector = np.sqrt(xdist**2 + ydist**2)
    print(f'calculated vector = {d_vector}<{np.degrees(angle)}')

    r = d_vector
    plt.polar(box_x, box_y ,'ro')
    plt.savefig(plot_save_name)
    
def stitching_task_YT_tutorial():
    image_paths = glob.glob('stitching/2/*.jpg')
    images=[]
    stitching_path = 'stitching/stitched_images_2'
    for image in image_paths:
        img=cv2.imread(image)
        images.append(img)
        cv2.imshow("image", img)
        cv2.waitKey(0)
    imageStitcher = cv2.Stitcher_create()
    error,stitched_img = imageStitcher.stitch(images)
    try:
        if not os.path.exists(stitching_path):
            os.makedirs(stitching_path)
    except IOError:
        print(f'unable to make or find the path {stitching_path} to place the stitched images')
    if not error:
        cv2.imwrite(stitching_path+'stitched_output.png', stitched_img)
        cv2.imshow('stitched image', stitched_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()




def stitching_task(): ###this could posibly only be done using the test.mp4
    #create array to store images
    imgs = []
    #create cv2 stitcher
    cv2.stitcher_create()
    #pick video to get frames as images
    vid_option = input('MAPPING: pick 0 to test with laptop camera, pick 1 to test with video.mp4, pick 2 to test with test.mp4')
    if vid_option == '1':
        videoCap = cv2.VideoCapture('video/video.mp4')
        print("starting stitching_task with video/video.mp4")
    elif vid_option == '2':
        videoCap = cv2.VideoCapture('video/test.mp4')
        print("starting stitching_task with video/test.mp4")
    elif vid_option == '0':
        videoCap = cv2.VideoCapture(0)
        print("starting stitching_task with laptop camera")
    else:
        videoCap = cv2.VideoCapture('video/video.mp4')
        print('invalid option starting stitching_task on the defaut video/video.mp4')
    current_frame = 0
    dir_name = './stitching/'+str(vid_option)
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    except OSError:
        print(f'error making the directory {dir_name}')
    while True:
        ret, frame = videoCap.read()
        if ret:
            if current_frame % 50 == 0:
                name = str(dir_name)+'/image_from_frame_'+str(current_frame)+'.jpg'
                print(f'saving frame as {name}')
                cv2.imwrite(name, frame)
            current_frame = current_frame + 1
        else:
            break
    videoCap.release()
    cv2.destroyAllWindows


task = 1
def main():
    if task ==1:
        object_detection()
    elif task == 2:
        stitching_task()
    elif task ==3:
        stitching_task_YT_tutorial()

if __name__ == "__main__":
    main()