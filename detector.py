import cv2
import mediapipe as mp


class EyeDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Right Eye Landmarks
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        # Left Eye Landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]

    def process_frame(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = frame.shape

                # Right Eye
                for idx in self.RIGHT_EYE:

                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)

                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                #