import numpy as np
import cv2
def print_feature(feature, frame):
    for (x, y, w, h) in feature:
        # print(x,y,w,h)
        roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y + h, x:x + w]

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
def wink_check(left_eye,right_eye, faces, frame):
    if(len(left_eye)+len(right_eye)==1):
        print_feature(faces, frame)
        if(len(left_eye)!=0):
            print("Left")
            print_feature(left_eye, frame)
        elif (len(right_eye)!=0):
            print("Right")
            print_feature(right_eye, frame)

face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Jhonny\Desktop\Personal_Projects\facial_recognition\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

right_eyes_cascade = cv2.CascadeClassifier(r'C:\Users\Jhonny\Desktop\Personal_Projects\facial_recognition\venv\Lib\site-packages\cv2\data\haarcascade_righteye_2splits.xml')
left_eyes_cascade = cv2.CascadeClassifier(r'C:\Users\Jhonny\Desktop\Personal_Projects\facial_recognition\venv\Lib\site-packages\cv2\data\haarcascade_lefteye_2splits.xml')
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    left_eye = left_eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    right_eye = right_eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    wink_check(left_eye, right_eye,faces, frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

