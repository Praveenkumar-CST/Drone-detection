import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet(" ", " ")  # Adjust file names if using YOLOv3
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# If "drone" is not in COCO, you can add it at a specific index
# For demonstration purposes, let's assume "drone" is already included in your model

# Initialize video capture
cap = cv2.VideoCapture(0)

def detect_drones(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    drone_detected = False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Ensure high confidence and that detected object is a drone
            if confidence > 0.5 and classes[class_id] == "drone":
                drone_detected = True
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "DRONE DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display alert if drone detected
    if drone_detected:
        print("!!!!!ALERT!!!! DRONE DETECTED")

    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect drones in frame
    frame = detect_drones(frame)
    cv2.imshow("Drone Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
