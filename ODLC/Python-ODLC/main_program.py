import pymavlink
import math
import numpy as np
from pymavlink.quaternion import QuaternionBase
from ultralytics import YOLO
import cv2
import time
import asyncio
import os
import threading
from pymavlink import mavutil
import Jetson.GPIO as GPIO
import orthophotos as mapping


def bbox_to_real_coord(x,y,x_center, y_center, height):
    pixels_to_make_meter = (0.256*1969)/height
    X= x-x_center
    Y = y-y_center
    return X,Y
def send_local_offset(master, x, y, z=0):
    master.mav.set_position_target_local_ned_send(
        int(time.time(1000)),
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b000111111000111,
        x,y,z,
        0,0,0,
        0,0,0,
        0,0
    )
def make_connection(port, baudrate):
    master = mavutil.mavlink_connection(port, baudrate, autoreconnect=True)
    master.wait_heartbeat()
    print('connected to system:', master.target_system)
    return master
geofence = {
    'latmin' : 84.4442,
    'latmax' : 85.3332,
    'lonmin' : 44.8898,
    'lonmax' : 44.7789
}
def geofence_check(lat, lon,geofence):
    return (geofence['latmin'] <= lat <= geofence['latmax'] and geofence['lonmin'] <= lon <= geofence['lonmax'])

def get_gps_position(master):
    msg = master.recv_match(type= 'Global_POSITION_INT', blocking=True, timeout=2)
    if msg:
        lat = msg.lat/1e7
        lon = msg.lon/1e7
        return lat, lon
    return None, None


ODLC_pause_event = threading.Event()
payload_delivered_event = threading.Event()
def object_detection(master):
    print('starting object detection')
    yolo = YOLO('visdrone100epoch.onnx')
    vid =True
    if not vid:
        cap = cv2.VideoCapture('video/video.mp4')
    else:
        cap = cv2.VideoCapture(0)
    scopecounter = 0
    while True:
        if not ODLC_pause_event.is_set():
            '''set ODLC_pause_event to halt'''
            ret,frame = cap.read()
            scopecounter = scopecounter+1
            if not ret:
                continue
            results = yolo.track(frame, stream=True, persist=False)
            for result in results:
                classes_names = result.names
                for box in result.boxes:
                    if box.conf > 0.8 and (scopecounter%100==0): # limit false positives
                        [x,y,w,h] = box.xywh[0]
                        x_center = frame.shape[1]/2
                        y_center= frame.shape[0]/2
                        X,Y = bbox_to_real_coord(x,y,x_center,y_center,15.24) #50ft = 15.24 meters
                        if np.abs(X) < 20 and np.abs(Y) <20: #make sure bounding box is close enough to the center of the image                 #well within bounds required by competition
                            payload_delivery(200,0.002)
                            payload_delivered_event.set()
                        else:
                            set_mode(master, True)
                            arm_drone(0)
                            send_local_offset(X,Y,0)
                            wait_until_positioned(master,20,0.2)
                
                    

def wait_until_positioned(master, timeout = 20, velocityThreshold = 0.2):
    print('waiting for the drone to hold a position')
    start = time.time()
    while time.time() - start<timeout:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg:
            vx = msg.vx / 100.0  # cm/s to m/s
            vy = msg.vy / 100.0
            speed = (vx**2 + vy**2)**0.5
            print(f"Speed: {speed:.2f} m/s")
            if speed < velocityThreshold:
                print("Drone is stable. Ready to run detection.")
                return True
    print("Timeout waiting for drone to hold position.")
    return False


def set_mode(master , ardupilot):
    if ardupilot == True:
        mode = 'GUIDED'
    else:
        mode = 'OFFBOARD'
    mode_id = master.mode_mapping()[mode]
    master.mav.set_mode_send(
        master.target_system,
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    #wait until mode has changed
    print(f'Waiting for mode change to {mode}...')
    while True:
        msg = master.recv_match(type = 'HEARTBEAT', blocking = True)
        current_mode_id = msg.custom_mode
        if current_mode_id == mode_id:
            print(f'Mode Changed to {mode}')
            break

def arm_drone(master):
    master.mav.command_long_send(
        master.target_system, 
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,1,0,0,0,0,0,0
    )

DIR_PIN = 32
STEP_PIN = 33
def setup_stepper():
    GPIO.setmode(GPIO.BOARD)
    time.sleep(1)
    GPIO.setup(DIR_PIN, GPIO.OUT)
    time.sleep(1)
    GPIO.setup(STEP_PIN, GPIO.OUT)
    time.sleep(1)

def move_stepper(dir, steps, delay):
    GPIO.output(DIR_PIN, GPIO.HIGH if dir else GPIO.LOW)
    unravellength = 5
    for step in range(steps*unravellength):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay)
    time.sleep(10)

def cleanup():
    GPIO.cleanup()

def payload_delivery(steps_in_motor, delay_in_seconds):                 
    setup_stepper()
    move_stepper(dir = True, steps = steps_in_motor, delay = delay_in_seconds)
    move_stepper(dir= False, steps= steps_in_motor, delay=delay_in_seconds)
    cleanup()

mapping_capture_done_event = threading.Event()
mapping_pause_event = threading.Event()
def mappingwrapper():
    cap = cv2.VideoCapture(0)
    img_dir = "images"
    os.makedirs(img_dir, exist_ok=True)
    framecounter = 0

    print("[Mapping] Starting image capture...")

    while not mapping_pause_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = int(time.time() * 1000)
        filename = f"{img_dir}/image_{framecounter}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        framecounter += 1
        time.sleep(1)

    cap.release()
    print("[Mapping] Capture stopped.")
    mapping_capture_done_event.set()

def MAPGen():
    mapping.generateMap()

initial_lap = True
detection_thread = None
mapping_thread = None
mapgen_thread = None    
def main():
    url = ''
    master= make_connection(url)
    while True:
        lat, lon = get_gps_position(master)
        if lat is None or lon is None:
            continue

        in_geofence = geofence_check(lat, lon, geofence)

        if in_geofence and initial_lap:
            print("[Main] Inside geofence, starting mapping...")
            ODLC_pause_event.set()  # block detection during mapping
            mapping_pause_event.clear()
            mapping_capture_done_event.clear()
            mapping_thread = threading.Thread(target=mappingwrapper, daemon=True)
            mapping_thread.start()
            initial_lap = False

        # Start ODLC once mapping is done
        if mapping_capture_done_event.is_set() and detection_thread is None:
            print("[Main] Starting object detection...")
            ODLC_pause_event.clear()
            detection_thread = threading.Thread(target=object_detection, args=(master,), daemon=True)
            detection_thread.start()

        # End ODLC if payload delivered or drone leaves geofence
        if (not in_geofence or payload_delivered_event.is_set()) and not ODLC_pause_event.is_set():
            print("[Main] Ending object detection...")
            ODLC_pause_event.set()
            payload_delivered_event.clear()

        # Once drone leaves geofence, trigger map generation
        if not in_geofence and mapping_capture_done_event.is_set() and mapgen_thread is None:
            print("[Main] Starting ODM map generation...")
            mapgen_thread = threading.Thread(target=generateMap, daemon=True)
            mapgen_thread.start()
            
            
               
               
    

                    
if __name__ =="__main__":
    main()