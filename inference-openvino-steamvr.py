from sys import platform
import socket
from ultralytics import YOLO

import cv2

cap = cv2.VideoCapture(0)

model = YOLO("yolo11m-pose_int8_openvino_model")

def close():
  pass

def sendToPipe(text):
    if platform.startswith('win32'):
        pipe = open(r'\\.\pipe\ApriltagPipeIn', 'rb+', buffering=0)
        some_data = str.encode(text)
        some_data += b'\0'
        pipe.write(some_data)
        resp = pipe.read(1024)
        pipe.close()
    elif platform.startswith('linux'):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.connect("/tmp/ApriltagPipeIn")
        some_data = text.encode('utf-8')
        some_data += b'\0'
        client.send(some_data)
        resp = client.recv(1024)
        client.close()
    else:
        print(f"Unsuported platform {platform}")
        raise Exception
    return resp

def sendToSteamVR(text):
    #Function to send a string to my steamvr driver through a named pipe.
    #open pipe -> send string -> read string -> close pipe
    #sometimes, something along that pipeline fails for no reason, which is why the try catch is needed.
    #returns an array containing the values returned by the driver.
    try:
        resp = sendToPipe(text)
    except:
        return ["error"]

    string = resp.decode("utf-8")
    array = string.split(" ")
    
    return array

def connect(smoothing, additional_smoothing):
        print("Connecting to SteamVR")

        #ask the driver, how many devices are connected to ensure we dont add additional trackers
        #in case we restart the program
        numtrackers = sendToSteamVR("numtrackers")
        if numtrackers is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            close()

        print(numtrackers)

        numtrackers = int(numtrackers[2])

        #games use 3 trackers, but we can also send the entire skeleton if we want to look at how it works
        totaltrackers = 3 # 23

        roles = ["TrackerRole_Waist", "TrackerRole_RightFoot", "TrackerRole_LeftFoot"]

        # if params.ignore_hip and not params.preview_skeleton:
        #     del roles[0]

        # if params.use_hands:
        #     roles.append("TrackerRole_Handed")
        #     roles.append("TrackerRole_Handed")

        for i in range(len(roles),totaltrackers):
            roles.append("None")

        for i in range(numtrackers,totaltrackers):
            #sending addtracker to our driver will... add a tracker. to our driver.
            resp = sendToSteamVR(f"addtracker MPTracker{i} {roles[i]}")
            if resp is None:
                print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
                close()

        resp = sendToSteamVR(f"settings 50 {smoothing} {additional_smoothing}")
        if resp is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            close()

def updatepose(params, pose3d, rots, hand_rots):
        array = sendToSteamVR("getdevicepose 0")        #get hmd data to allign our skeleton to

        if array is None or len(array) < 10:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            close(params)

        headsetpos = [float(array[3]),float(array[4]),float(array[5])]
        headsetrot = R.from_quat([float(array[7]),float(array[8]),float(array[9]),float(array[6])])

        neckoffset = headsetrot.apply(params.hmd_to_neck_offset)   #the neck position seems to be the best point to allign to, as its well defined on
                                                            #the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.

        if params.recalibrate:
            print("INFO: frame to recalibrate")

        else:
            pose3d = pose3d * params.posescale     #rescale skeleton to calibrated height
            #print(pose3d)
            offset = pose3d[7] - (headsetpos+neckoffset)    #calculate the position of the skeleton
            if not params.preview_skeleton:
                numadded = 3
                if not params.ignore_hip:
                    for i in [(0,1),(5,2),(6,0)]:
                        joint = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {params.camera_latency} 0.8")
                else:
                    for i in [(0,1),(5,2)]:
                        joint = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {params.camera_latency} 0.8")
                        numadded = 2
                if params.use_hands:
                    for i in [(10,0),(15,1)]:
                        joint = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(f"updatepose {i[1]+numadded} {joint[0]} {joint[1]} {joint[2]} {hand_rots[i[1]][3]} {hand_rots[i[1]][0]} {hand_rots[i[1]][1]} {hand_rots[i[1]][2]} {params.camera_latency} 0.8")
            else:
                for i in range(23):
                    joint = pose3d[i] - offset      #if previewing skeleton, send the position of each keypoint to steamvr without rotation
                    sendToSteamVR(f"updatepose {i} {joint[0]} {joint[1]} {joint[2] - 2} 1 0 0 0 {params.camera_latency} 0.8")
        return True

connect(0, 0)

