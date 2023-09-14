import cv2
import sys

faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        minNeighbors=10,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:

        nx = int(x + 0.1 * w)
        nw = int(0.4 * w)
        ny = int(y + 0.2 * h)
        nh = int(0.3 * h)

        cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

        eyes = eyeCascade.detectMultiScale(
            frame[ny: ny + nh, nx: nx + nw],
            minNeighbors=10
        )
        # draw a rectangle around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (nx + ex, ny + ey), (nx + ex + ew, ny + ey + eh), (0, 255, 255), 2)

            eye1 = gray[ny + ey: ny + ey + eh, nx + ex: nx + ex + ew]
            ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('Video', binary)

    # Display the resulting frame
    cv2.imshow('Video', frame)  # entire frame
    # cv2.imshow('Video', frame)  # eye only


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
