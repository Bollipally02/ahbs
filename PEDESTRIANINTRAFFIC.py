import numpy as np
import cv2
import os
import imutils

# detect pedestrians in the image and return the bounding boxes and centroids of the detected pedestrians
def all_types_of_pedestrian_detection(image, machine_model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []

	# construct a blob from the input image and then perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	machine_model.setInput(blob)
	detections = machine_model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	# loop over each of the layer outputs and extract the bounding boxes and associated probabilities
	for output in detections:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > 0.2:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
	if len(idzs) > 0:
		for i in idzs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	return results

# initialize the list of class labels our YOLO model was trained to detect
LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# load our YOLO object detector trained on COCO dataset (80 classes)
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

machine_model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

'''
machine_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
machine_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''

layer_name = machine_model.getLayerNames()
layer_name = [layer_name[i-1] for i in machine_model.getUnconnectedOutLayers()]
# capturing the video from the file
capture = cv2.VideoCapture("person1.mp4")

# loop over the frames from the video stream
while True:
	(grabbed, image) = capture.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	image = imutils.resize(image, width=700)

	# detect pedestrians in the image and return the bounding boxes and centroids of the detected pedestrians
	results = all_types_of_pedestrian_detection(image, machine_model, layer_name,
		personidz=LABELS.index("person"))
	
	# get the number of detected people in this frame
	count = len(results)

	# create a label with the count and some formatting
	label = f"Detected: {count}"

	# draw the label on the frame color red
	cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
	for res in results:
		cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

		# draw the centroid of the object on the output frame
	cv2.imshow("Pedestrian Detection", image)

	# if the `q` key was pressed, break from the loop
	key = cv2.waitKey(1)

	# if 'esc' key or window is closed, break from the loop
	if key == 27 or cv2.getWindowProperty("Pedestrian Detection", cv2.WND_PROP_AUTOSIZE) != cv2.WND_PROP_AUTOSIZE:
		break

# release the file pointers
capture.release()
# close any open windows
cv2.destroyAllWindows()