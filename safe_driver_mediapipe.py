import cv2
import mediapipe as mp
import pygame
import math

# -------------------- Alarm Setup --------------------
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Place alarm.wav in the same folder

# -------------------- Mediapipe Setup --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- Thresholds --------------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 90   # ~3 seconds at 30fps
MOUTH_AR_THRESH = 0.7

# -------------------- Counters --------------------
eye_counter = 0

# Eye & Mouth landmark indexes (Mediapipe face mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [61, 291, 81, 178, 13, 14]

# -------------------- Helper Functions --------------------
def euclidean(pt1, pt2):
    return math.dist(pt1, pt2)

def eye_aspect_ratio(landmarks, eye):
    A = euclidean(landmarks[eye[1]], landmarks[eye[5]])
    B = euclidean(landmarks[eye[2]], landmarks[eye[4]])
    C = euclidean(landmarks[eye[0]], landmarks[eye[3]])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth):
    A = euclidean(landmarks[mouth[2]], landmarks[mouth[5]])
    B = euclidean(landmarks[mouth[3]], landmarks[mouth[4]])
    C = euclidean(landmarks[mouth[0]], landmarks[mouth[1]])
    return (A + B) / (2.0 * C)

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    alert_text = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

            # Eye Aspect Ratio (EAR)
            leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
            rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (leftEAR + rightEAR) / 2.0

            # Mouth Aspect Ratio (MAR)
            mar = mouth_aspect_ratio(landmarks, MOUTH)

            # Eye closure detection
            if ear < EYE_AR_THRESH:
                eye_counter += 1
                if eye_counter >= EYE_AR_CONSEC_FRAMES:
                    alert_text = "ALERT: Eyes closed for too long!"
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
            else:
                eye_counter = 0

            # Yawning detection
            if mar > MOUTH_AR_THRESH:
                alert_text = "ALERT: Yawning detected!"
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)

    # Stop alarm if no alert
    if alert_text == "":
        pygame.mixer.music.stop()

    # Display alert text
    if alert_text != "":
        cv2.putText(frame, alert_text, (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Safe Driver", frame)

    # Quit with "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()


