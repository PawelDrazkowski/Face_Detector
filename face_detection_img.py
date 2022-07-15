import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("faces.png")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale for better face detection accuracy

faces = face_cascade.detectMultiScale(gray_img, 
scaleFactor = 1.05, 
minNeighbors = 10)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)

scale = 1
resized_img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))

cv2.imshow("Detected faces", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        