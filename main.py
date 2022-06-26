import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)


# load our serialized face detector model
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model
maskNet = load_model("mask_detector.model")

# load to focus on the structure or the shape of an object
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
line_length_prev = 0
line_length_curr = 0

# initialize the video stream
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('captured.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 8, (frame_width, frame_height))


while True:

    key = cv2.waitKey(1) & 0xFF
    successful_frame_read, frame = cap.read()

    # detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # detects the distance between people
    try:
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    except:
        pass
    else:
        for (x, y, w, h) in boxes:
            if w > 110:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (x+int(w/2), y+int(h/2)), 3, (0, 0, 255), 3)
        if len(boxes) > 1:
            x1, y1, w1, h1 = boxes[0]
            x2, y2, w2, h2 = boxes[1]

            if w1 > 110 and w2 > 110:
                line_length_prev = line_length_curr
                line_length_curr = int(
                    ((x1+(w1/2)-x2-(w2/2))**2 + (y1+(h1/2)-y2-(h2/2))**2)**0.5)
                if line_length_curr < line_length_prev/2:
                    continue
                cv2.line(frame, (x1+int(w1/2), y1+int(h1/2)),
                         (x2+int(w2/2), y2+int(h2/2)), (255, 255, 0), 4)
                min = (h2, h1)[h1 < h2]
                max = (h1, h2)[h1 < h2]
                cv2.putText(frame, "Distance Between Them: " + str(line_length_curr),
                            (40, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 3)
                if line_length_curr < max:
                    cv2.putText(frame, 'ALERT! Too Close!', (350, 200),
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
    finally:
        if successful_frame_read == True:
            cv2.imshow('frame', frame)
            out.write(frame)

    # Stops code if 'S' key is pressed
    if key == 83 or key == 115:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
