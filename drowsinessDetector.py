import cv2
import dlib
import pygame
from scipy.spatial import distance
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
    print("Drowsy")
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

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

aver_nose_face = []
nose_face = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        nose = []
        face = []

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

        #alarm system
        if EAR<0.26:
            init_message()
        elif incl > 0.5:
            init_message()
        elif nose_face - average > 0.05:
            init_message()
        else:
            onalarm = False

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()