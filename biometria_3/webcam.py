import cv2
import torch
import logging
from PIL import Image
from config import Config

config = Config()

# ImageNet normalization for pretrained models
transform = config.transform


def webcam_face_detection(model, device, confidence_threshold=0.5, enable_fps_display=True, mirror=True):
    """Live webcam face detection"""
    import time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam")
        return

    model.eval()
    fps_history = []

    logging.info("Live YOLO Face Detection Started")
    logging.info("Press 'q' to quit, 'p' to pause/resume")

    paused = False

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture image from webcam")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        display_frame = frame.copy()

        if not paused:
            start_time = time.time()

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Perform detection
            with torch.no_grad():
                predictions = model(input_tensor)

            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            # Draw bounding boxes
            faces_detected = 0

            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)

                    faces_detected += 1
                    color = (0, 255, 0)  # Green for faces

                    # Draw rectangle and label
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"Face: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display status
            status_text = f"{faces_detected} Face(s) Detected" if faces_detected > 0 else "No Faces Detected"
            status_color = (0, 255, 0) if faces_detected > 0 else (0, 0, 255)
            cv2.putText(display_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # FPS display
            if enable_fps_display:
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "PAUSED (Press 'p' to resume)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Instructions
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('YOLO Face Detection', display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    logging.info("YOLO face detection stopped")


def ask_for_webcam_inference(model, config, device):
    """Pyta użytkownika czy uruchomić detekcję twarzy z kamery"""
    print("\nDo you want to start live webcam face detection? (y/n)")
    choice = input().strip().lower()
    if choice in ['y', 'yes']:
        logging.info("User chose to start live YOLO face detection.")
        webcam_face_detection(
            model=model,
            device=device,
            confidence_threshold=config.face_confidence_threshold,
            enable_fps_display=True,
            mirror=True
        )
    else:
        logging.info("User declined to start live YOLO face detection.")
