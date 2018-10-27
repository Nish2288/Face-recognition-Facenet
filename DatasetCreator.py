# Face Recognition

# Importing the libraries
import cv2
import os

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create folders to store images.
def capture_name(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
name=input('Enter Name:')
capture_name("webcam/"+str(name))

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 15)
    cv2.putText(frame, "Press c to capture image.", (10,50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    for (x, y, w, h) in faces:        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    k=cv2.waitKey(1)
    if k==99:
        
        cv2.imwrite("webcam/"+str(name)+"/"+name+'.jpg',frame[y:y+h,x:x+w])
        print('Image captured successfully')
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
video_capture.release()
cv2.destroyAllWindows()
