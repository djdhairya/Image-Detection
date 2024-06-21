import cv2
from ultralytics import YOLO
import supervision as sv

frame_width = 1280
frame_height = 720

def detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    model = YOLO("yolov8l.pt")
    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detection()