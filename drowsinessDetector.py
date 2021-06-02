import cv2
import dlib
import pygame
import time
from scipy.spatial import distance
from datetime import datetime

import numpy as np

#check if alarm is on
onalarm = False

def sound_alarm(path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def init_message():
    global onalarm

    cv2.putText(frame, "Drowsy", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
    cv2.putText(frame, "Are you Sleepy?", (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    # print("Drowsy")
    if onalarm == False :
        onalarm = True
        sound_alarm("alarm.mp3")

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def calculate_NOSE(nose):
    # print(distance.euclidean(nose[0],nose[-1]))
    return distance.euclidean(nose[0],nose[-1])
def calculate_FACE(face):
    # print(distance.euclidean(face[0],face[1]))
    return distance.euclidean(face[0],face[-1])

#안경 판별 함수들
def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def get_centers(img, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_OUTTER[0] + EYE_LEFT_INNER[0]) / 2
    x_right = (EYE_RIGHT_OUTTER[0] + EYE_RIGHT_INNER[0]) / 2
    LEFT_EYE_CENTER = np.array([np.int32(x_left), np.int32(x_left * k + b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right * k + b)])

    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)
    cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyescenter = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx * dx + dy * dy)
    scale = desired_dist / dist
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))

    return aligned_face


def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    # cv2.imshow('sobel_y', sobel_y)

    edgeness = sobel_y

    retVal, thresh = cv2.threshold(edgeness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    d = len(thresh) * 0.5
    x = np.int32(d * 6 / 7)
    y = np.int32(d * 3 / 4)
    w = np.int32(d * 2 / 7)
    h = np.int32(d * 2 / 4)

    x_2_1 = np.int32(d * 1 / 4)
    x_2_2 = np.int32(d * 5 / 4)
    w_2 = np.int32(d * 1 / 2)
    y_2 = np.int32(d * 8 / 7)
    h_2 = np.int32(d * 1 / 2)

    roi_1 = thresh[y:y + h, x:x + w]  
    roi_2_1 = thresh[y_2:y_2 + h_2, x_2_1:x_2_1 + w_2]
    roi_2_2 = thresh[y_2:y_2 + h_2, x_2_2:x_2_2 + w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure_1 = sum(sum(roi_1 / 255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])  
    measure_2 = sum(sum(roi_2 / 255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1]) 
    measure = measure_1 * 0.3 + measure_2 * 0.7

    # cv2.imshow('roi_1', roi_1)
    # cv2.imshow('roi_2', roi_2)
    # print(measure)

    if measure > 0.15:
        judge = True
    else:
        judge = False
    # print(judge)
    return judge

def getting_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(1, 2, 255))
        cv2.circle(im, pos, 3, color=(0, 2, 2))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])

def mouth_open(image):
    landmarks = getting_landmarks(image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

aver_nose_face = []
nose_face = 0
# 시간 측정을 위한 리스트
EAR_time =[]
incl_time =[]
nose_face_time =[]

#안경 예측
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

global yawn_start_time
global yawn_finish_time

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    image_landmarks, lip_distance = mouth_open(frame)

    prev_yawn_status = yawn_status

    if lip_distance > 50:
        yawn_status = True

        cv2.putText(frame, "Yawning", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        output_text = " Yawn Count : " + str(yawns + 1)

        cv2.putText(frame, output_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

    else:
        yawn_status = False

    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
    
    cv2.imshow('Live Landmarks', image_landmarks)
    cv2.imshow('Yawn Detection', frame)

    if yawns == 1:
        yawn_start_time = time.time()
        # print(yawn_start_time)

    if yawns == 5:
        yawn_finish_time = time.time()
        print(yawn_finish_time - yawn_start_time)
        if yawn_finish_time - yawn_start_time == 120:
            init_message()
        yawns = 0


    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        nose = []
        face = []
        mouth = []
        
        # landmarks = predictor(gray, face)
        #nose
        x = face_landmarks.part(27).x
        y = face_landmarks.part(27).y
        nose.append((x, y))
        x = face_landmarks.part(33).x
        y = face_landmarks.part(33).y
        nose.append((x, y))
        #face
        x = face_landmarks.part(27).x
        y = face_landmarks.part(27).y
        face.append((x, y))
        x = face_landmarks.part(8).x
        y = face_landmarks.part(8).y
        face.append((x, y))
        nose_face = calculate_NOSE(nose)/calculate_FACE(face)
        if len(aver_nose_face)< 30:
            aver_nose_face.append(nose_face)

        average = np.mean(np.array(aver_nose_face))

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)

		# eye inclination check
        x1 = face_landmarks.part(39).x - face_landmarks.part(36).x
        y1 = face_landmarks.part(39).y - face_landmarks.part(36).y
        incl1 = y1 / x1

        x2 = face_landmarks.part(45).x - face_landmarks.part(42).x
        y2 = face_landmarks.part(45).y - face_landmarks.part(42).y
        incl2 = y2 / x2

        incl = abs((incl1+incl2)/2)
        rects = detector(gray, 1)

        for i, rect in enumerate(rects):

            x_face = rect.left()
            y_face = rect.top()
            w_face = rect.right() - x_face
            h_face = rect.bottom() - y_face

            cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
            cv2.putText(frame, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

            landmarks = predictor(gray, rect)
            landmarks = landmarks_to_np(landmarks)
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                # print((x,y))    

            LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(frame, landmarks)

            aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
            cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)

            judge = judge_eyeglass(aligned_face)
            #judge 안경 유무
            if judge == True:
                cv2.putText(frame, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2,
                            cv2.LINE_AA)
                #alarm system
                # 눈부분
                if EAR < 0.24:
                    if len(EAR_time)>0:
                        if (datetime.now()-EAR_time[0]).seconds > 1:
                            init_message()
                            print("EAR")
                    else:
                        EAR_time.append(datetime.now())
                elif len(EAR_time)>0:
                    EAR_time.pop()

                if incl > 0.5:
                    if len(incl_time) > 0:
                        if (datetime.now() - incl_time[0]).seconds > 1:
                            init_message()
                            print("incl")
                    else:
                        incl_time.append(datetime.now())
                elif len(incl_time) > 0:
                    incl_time.pop()

                if nose_face - average > 0.05:
                    if len(nose_face_time) > 0:
                        if (datetime.now() - nose_face_time[0]).seconds > 2:
                            init_message()
                            print("nose_face")
                    else:
                        nose_face_time.append(datetime.now())
                elif len(nose_face_time) > 0:
                    nose_face_time.pop()
                if not(EAR<0.20) and not(incl > 0.5) and not(nose_face - average > 0.05):
                    onalarm = False
            else:
                # alarm system
                # 눈부분
                if EAR < 0.26:
                    if len(EAR_time) > 0:
                        if (datetime.now() - EAR_time[0]).seconds > 1:
                            init_message()
                            print("EAR")
                    else:
                        EAR_time.append(datetime.now())
                elif len(EAR_time) > 0:
                    EAR_time.pop()

                if incl > 0.5:
                    if len(incl_time) > 0:
                        if (datetime.now() - incl_time[0]).seconds > 1:
                            init_message()
                            print("incl")

                    else:
                        incl_time.append(datetime.now())
                elif len(incl_time) > 0:
                    incl_time.pop()

                if nose_face - average > 0.05:
                    if len(nose_face_time) > 0:
                        if (datetime.now() - nose_face_time[0]).seconds > 2:
                            init_message()
                            print("nose_face")
                    else:
                        nose_face_time.append(datetime.now())
                elif len(nose_face_time) > 0:
                    nose_face_time.pop()
                if not (EAR < 0.20) and not (incl > 0.5) and not (nose_face - average > 0.05):
                    onalarm = False

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()






# cap = cv2.VideoCapture(0)
#
# while (cap.isOpened()):
#     _, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     rects = detector(gray, 1)
#
#     for i, rect in enumerate(rects):
#
#         x_face = rect.left()
#         y_face = rect.top()
#         w_face = rect.right() - x_face
#         h_face = rect.bottom() - y_face
#
#         cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
#         cv2.putText(frame, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                     (0, 255, 0), 2, cv2.LINE_AA)
#
#         landmarks = predictor(gray, rect)
#         landmarks = landmarks_to_np(landmarks)
#         for (x, y) in landmarks:
#             cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
#
#         LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
#
#         aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
#         cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)
#
#         judge = judge_eyeglass(aligned_face)
#         if judge == True:
#             cv2.putText(frame, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
#                         cv2.LINE_AA)
#
#
#     cv2.imshow("Result", frame)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()