import cv2
import mediapipe as mp
import numpy as np
from controlkeys import right_pressed, left_pressed, up_pressed, down_pressed
from controlkeys import KeyOn, KeyOff
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Mediapipe Hand Detector
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

# Define tip IDs for fingers
tipIds = np.array([4, 8, 12, 16, 20])
# Define IDs for second joints (for comparison)
secondJointIds = np.array([2, 6, 10, 14, 18])

def get_hand_label(results, index=0):
    """
    Retrieve the hand label and return the actual direction for controls.
    MediaPipe labels are from the camera's POV, so we need to flip them
    for the user's POV.
    """
    try:
        label = results.multi_handedness[index].classification[0].label
        # Flip the label because MediaPipe gives camera POV
        return "Left" if label == "Right" else "Right"
    except IndexError:
        return None

def landmarks_to_array(hand_landmarks, image_shape):
    """Convert hand landmarks to numpy array for faster processing."""
    h, w = image_shape[:2]
    landmarks = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark])
    return landmarks

def calculate_fingers(landmarks):
    """Calculate which fingers are up using numpy operations."""
    if len(landmarks) < 21:  # Ensure we have all landmarks
        return []
    
    # Convert landmarks to numpy array if not already
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    
    fingers = np.zeros(5, dtype=np.int32)
    
    # Thumb (comparing x-coordinates)
    fingers[0] = 1 if landmarks[tipIds[0], 0] > landmarks[secondJointIds[0], 0] else 0
    
    # Other fingers (comparing y-coordinates)
    # Get y-coordinates for fingertips and second joints
    tips_y = landmarks[tipIds[1:], 1]
    second_joints_y = landmarks[secondJointIds[1:], 1]
    
    # Compare positions (finger is up if tip is higher than second joint)
    fingers[1:] = (tips_y < second_joints_y).astype(np.int32)
    
    return fingers

def main():
    # Track current pressed keys
    current_key_pressed = set()
    
    # Webcam feed
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        logger.error("Failed to open webcam")
        return

    with mp_hand.Hands(min_detection_confidence=0.8,
                    min_tracking_confidence=0.7,
                    max_num_hands=2) as hands:
        while True:
            ret, image = video.read()
            if not ret:
                logger.error("Failed to read frame")
                break

            # Preprocess image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for idx, hand_landmark in enumerate(results.multi_hand_landmarks):
                    # Convert landmarks to numpy array
                    landmarks = landmarks_to_array(hand_landmark, image.shape)
                    
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
                    
                    # Get hand label (now correctly flipped)
                    hand_label = get_hand_label(results, idx)
                    logger.debug(f"Detected hand: {hand_label}")

                    # Calculate fingers
                    fingers = calculate_fingers(landmarks)
                    total_fingers = np.sum(fingers)
                    logger.debug(f"Fingers up: {fingers}, Total: {total_fingers}")

                    # Release all currently pressed keys first
                    for key in current_key_pressed.copy():
                        KeyOff(key)
                        current_key_pressed.remove(key)
                        logger.debug(f"Released key: {key}")

                    try:
                        if fingers[1] == 1 and total_fingers == 1:  # Index finger only
                            if hand_label == "Right":
                                logger.debug("Pressing D key (Right)")
                                KeyOn(right_pressed)
                                current_key_pressed.add(right_pressed)
                                cv2.putText(image, "RIGHT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            elif hand_label == "Left":
                                logger.debug("Pressing A key (Left)")
                                KeyOn(left_pressed)
                                current_key_pressed.add(left_pressed)
                                cv2.putText(image, "LEFT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        elif total_fingers == 5:  # All fingers up
                            logger.debug("Pressing W key (Up)")
                            KeyOn(up_pressed)
                            current_key_pressed.add(up_pressed)
                            cv2.putText(image, "UP", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        elif total_fingers == 0:  # All fingers down
                            logger.debug("Pressing S key (Down)")
                            KeyOn(down_pressed)
                            current_key_pressed.add(down_pressed)
                            cv2.putText(image, "DOWN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    except Exception as e:
                        logger.error(f"Error in key control: {str(e)}")

            # Release keys if no hands detected
            if not results.multi_hand_landmarks:
                for key in current_key_pressed.copy():
                    KeyOff(key)
                    current_key_pressed.remove(key)
                    logger.debug("No hands detected, releasing all keys")

            # Display the webcam feed
            cv2.imshow("Frame", image)

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()