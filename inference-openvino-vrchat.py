from ultralytics import YOLO
import cv2
from pythonosc import udp_client
from torch import Tensor

OSC_IP = "127.0.0.1"
OSC_PORT = 9000
OSC_PREFIX = "/avatar/parameters"
CLAMP_RANGE = True

client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

cap = cv2.VideoCapture(0)

model = YOLO("yolo11n-pose_int8_openvino_model")

# LANDMARK_TO_PARAM = {
#     mp_pose.PoseLandmark.LEFT_HIP:  "LeftHip",
#     mp_pose.PoseLandmark.RIGHT_HIP: "RightHip",
#     mp_pose.PoseLandmark.LEFT_KNEE: "LeftKnee",
#     mp_pose.PoseLandmark.RIGHT_KNEE:"RightKnee",
#     mp_pose.PoseLandmark.LEFT_ANKLE:"LeftAnkle",
#     mp_pose.PoseLandmark.RIGHT_ANKLE:"RightAnkle",

#     mp_pose.PoseLandmark.LEFT_SHOULDER: "LeftShoulder",
#     mp_pose.PoseLandmark.RIGHT_SHOULDER:"RightShoulder",
#     mp_pose.PoseLandmark.LEFT_ELBOW:    "LeftElbow",
#     mp_pose.PoseLandmark.RIGHT_ELBOW:   "RightElbow",
#     mp_pose.PoseLandmark.LEFT_WRIST:    "LeftWrist",
#     mp_pose.PoseLandmark.RIGHT_WRIST:   "RightWrist",

#     mp_pose.PoseLandmark.NOSE:    "Head",
#     mp_pose.PoseLandmark.LEFT_EYE: "LeftEye",
#     mp_pose.PoseLandmark.RIGHT_EYE:"RightEye",
#     mp_pose.PoseLandmark.LEFT_EAR: "LeftEar",
#     mp_pose.PoseLandmark.RIGHT_EAR:"RightEar",

#     # torso / spine approximations
#     mp_pose.PoseLandmark.LEFT_SHOULDER: "SpineLeft",
#     mp_pose.PoseLandmark.RIGHT_SHOULDER:"SpineRight",
#     mp_pose.PoseLandmark.LEFT_HIP: "HipsLeft",
#     mp_pose.PoseLandmark.RIGHT_HIP:"HipsRight",
# }

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def send_param_float(param_base, axis, value):
    """
    Send a single float OSC to VRChat:
      e.g. /avatar/parameters/LeftWristX  <float>
    """
    name = f"{OSC_PREFIX}/{param_base}{axis}"
    try:
        if CLAMP_RANGE:
            value = clamp(value, 0.0, 1.0)
        client.send_message(name, float(value))
    except Exception as e:
        # don't crash on network hiccups; print once.
        print(f"[OSC send error] {name} -> {value} : {e}")

def send_keypoint(keypoint: Tensor, name):
    x = keypoint[0].item()
    y = keypoint[y].item()
    print(name, x, y)
    send_param_float(name, "X", x)
    send_param_float(name, "Y", y)


while True:
    # send_param_float("Head", "X", .5)
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 480))

    results = model.predict(img)[0]
    keypoint = results.keypoints.xyn[0][0]
    print(keypoint[0], keypoint[1])
    send_keypoint(keypoint, "Head")

    print(len(results[0]))
    # print(results[0][0])
    img = results.plot()

    cv2.imshow("frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


