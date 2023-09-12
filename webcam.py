import cv2
import sys

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        minNeighbors=5,
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
            frame[ny: ny + nh, nx : nx + nw],
            minSize=(80, 80),
            maxSize=(100, 100)
        )
        # draw a rectangle around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (nx + ex, ny + ey), (nx + ex + ew, ny + ey + eh), (0, 255, 255), 2)

            eye1 = gray[y + ey: y + ey + eh, x + ex: x + ex + ew]
            ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('Video', binary)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# import cv2
#
# # read input image
# video_capture = cv2.VideoCapture(1)
#
# # read the haarcascade to detect the faces in an image
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# # read the haarcascade to detect the eyes in an image
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#
#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()
#
# # detects faces in the input image
# faces = face_cascade.detectMultiScale(gray, 1.3, 4)
# print('Number of detected faces:', len(faces))
#
# # loop over the detected faces
# for (x, y, w, h) in faces:
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = img[y:y + h, x:x + w]
#
#     # detects eyes of within the detected face area (roi)
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#
#     # draw a rectangle around eyes
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
#
# # display the image with detected eyes
# cv2.imshow('Eyes Detection', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
