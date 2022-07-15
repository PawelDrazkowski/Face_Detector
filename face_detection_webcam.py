import cv2 

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_detection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, 
    scaleFactor = 1.3, 
    minNeighbors = 5)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)

    return img

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    img = cv2.flip(frame, flipCode=1)

    cv2.imshow("Capture", face_detection(img))

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()