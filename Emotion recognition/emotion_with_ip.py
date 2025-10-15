from facial_emotion_recognition import EmotionRecognition
import cv2
import imutils

# Initialize the Emotion Recognition model
er = EmotionRecognition(device='cpu')

# Open webcam (0 = default camera)
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Recognize emotion
    frame = er.recognise_emotion(frame, return_type='BGR')

    # Resize for better display
    frame = imutils.resize(frame, width=450)

    # Show output
    cv2.imshow("Facial Emotion Recognition", frame)

    # Exit when 'ESC' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release and close
webcam.release()
cv2.destroyAllWindows()

