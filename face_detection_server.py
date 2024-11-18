import cv2
import asyncio
import websockets
import numpy as np
import mediapipe as mp
import base64
import json

# WebSocket server port
PORT = 6789

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

async def send_proximity_and_gesture(websocket):
    # Start the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("WebSocket server started. Waiting for connection...")

    prev_x = None
    gesture_command = None

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection
            face_results = face_detection.process(rgb_frame)

            # Hand detection
            hand_results = hands.process(rgb_frame)

            # Draw face detection annotations on the image
            proximity = "No Face Detected"
            if face_results.detections:
                for detection in face_results.detections:
                    mp_draw.draw_detection(frame, detection)
                    # Extract bounding box size
                    bbox = detection.location_data.relative_bounding_box
                    bbox_width = bbox.width

                    # Decide proximity based on bounding box width
                    if bbox_width > 0.3:  # Adjust threshold as needed
                        proximity = "Close Proximity"
                    else:
                        proximity = "Far Away"
                    break  # Only consider the first face detected

            # Draw hand landmarks on the image
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the x-coordinate of the index finger tip
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    current_x = index_finger_tip.x  # Normalized x-coordinate (0 to 1)

                    if prev_x is not None:
                        delta_x = current_x - prev_x

                        # Define a threshold to detect significant movement
                        movement_threshold = 0.05

                        if delta_x > movement_threshold:
                            gesture_command = "Swipe Right"
                        elif delta_x < -movement_threshold:
                            gesture_command = "Swipe Left"

                    prev_x = current_x
                    break  # Only consider the first hand detected
            else:
                prev_x = None
                gesture_command = None

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Create a JSON message containing the image and data
            message = {
                'type': 'frame',
                'image': jpg_as_text,
                'proximity': proximity,
                'gesture': gesture_command
            }

            # Send the JSON message to the client
            await websocket.send(json.dumps(message))

            # Reset gesture command after sending
            gesture_command = None

            # Wait a short time before sending the next update
            await asyncio.sleep(0.033)  # Approx 30 frames per second
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cap.release()

async def main():
    print(f"Starting WebSocket server on ws://0.0.0.0:{PORT}")
    async with websockets.serve(send_proximity_and_gesture, "0.0.0.0", PORT):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("WebSocket server stopped.")
