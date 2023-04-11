import cv2
import time

face_cascade = cv2.CascadeClassifier("haarcascade_front.xml")

video = cv2.VideoCapture(0)

prev_frame_time = 0
 
new_frame_time = 0

while True:
    check,frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=9)

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps) 

    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()