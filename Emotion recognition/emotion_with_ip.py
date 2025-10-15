from facial_emotion_recognition import EmotionRecognition
import urllib.request
import cv2
import numpy as np
import imutils

# Initialize the Emotion Recognition model
er = EmotionRecognition(device='cpu')

# IP camera URL (from mobile IP Webcam app)
url = 'http://192.168.1.33:8080/shot.jpg'

while True:
    # Read image from the IP camera
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)

    # Perform emotion recognition
    frame = er.recognise_emotion(frame, return_type='BGR')

    # Resize frame for better display
    frame = imutils.resize(frame, width=450)

    # Display the result
    cv2.imshow("Frame", frame)

    # Exit when 'ESC' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()


