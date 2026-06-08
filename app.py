import math
import tkinter as tk
import time
from collections import deque

import cv2
import customtkinter as ctk
import mediapipe as mp
from PIL import Image, ImageTk


class BlinkDetector:
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_DETAIL = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]
    LEFT_EYE_DETAIL = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.closed_threshold = 0.20
        self.open_threshold = 0.24
        self.is_closed = False
        self.closed_frames = 0
        self.blink_count = 0
        self.blink_times = deque(maxlen=30)
        self.open_ear_samples = deque(maxlen=45)
        self.session_start = time.time()
        self.lowest_ear = 1.0
        self.last_face_seen = 0

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _eye_aspect_ratio(self, points):
        left, top1, top2, right, bottom1, bottom2 = points
        vertical = self._distance(top1, bottom1) + self._distance(top2, bottom2)
        horizontal = 2.0 * max(self._distance(left, right), 1.0)
        return vertical / horizontal

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        metrics = {
            "face_detected": False,
            "blink_count": self.blink_count,
            "frequency": 0.0,
            "heart_rate": 0,
            "confidence": 0.0,
        }

        if not results.multi_face_landmarks:
            return metrics

        self.last_face_seen = time.time()
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = face_landmarks.landmark

        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
        y1, y2 = max(min(ys), 0), min(max(ys), h - 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Face Detected",
            (x1 + 40, max(y1 - 12, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        right_eye_points = self._landmark_points(landmarks, self.RIGHT_EYE, w, h)
        left_eye_points = self._landmark_points(landmarks, self.LEFT_EYE, w, h)

        for idx in self.RIGHT_EYE_DETAIL + self.LEFT_EYE_DETAIL:
            point = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        right_ear = self._eye_aspect_ratio(right_eye_points)
        left_ear = self._eye_aspect_ratio(left_eye_points)
        ear = (right_ear + left_ear) / 2.0
        self._update_thresholds(ear)

        self.lowest_ear = min(self.lowest_ear, ear)
        if ear < self.closed_threshold:
            self.closed_frames += 1
            if self.closed_frames >= 1:
                self.is_closed = True
        elif ear > self.open_threshold:
            if self.is_closed and self.closed_frames >= 1:
                self.blink_count += 1
                self.blink_times.append(time.time())
            self.is_closed = False
            self.closed_frames = 0
            self.lowest_ear = ear
        else:
            if self.is_closed and self.lowest_ear < self.open_threshold * 0.92 and ear > self.open_threshold * 0.96:
                self.blink_count += 1
                self.blink_times.append(time.time())
                self.is_closed = False
                self.closed_frames = 0
                self.lowest_ear = ear

        now = time.time()
        recent_blinks = [t for t in self.blink_times if now - t <= 60]
        elapsed_minutes = max((now - self.session_start) / 60.0, 1 / 60.0)
        frequency = self.blink_count / max(now - self.session_start, 1.0)
        blink_rate = self.blink_count / elapsed_minutes
        heart_rate = self._estimate_heart_rate(blink_rate)
        confidence = min(99.0, max(40.0, (ear / self.open_threshold) * 80.0))

        metrics.update(
            {
                "face_detected": True,
                "blink_count": self.blink_count,
                "frequency": frequency,
                "heart_rate": heart_rate,
                "confidence": confidence,
                "ear": ear,
            }
        )
        return metrics

    def _landmark_points(self, landmarks, indexes, width, height):
        return [(int(landmarks[idx].x * width), int(landmarks[idx].y * height)) for idx in indexes]

    def _update_thresholds(self, ear):
        if self.is_closed or ear <= 0:
            return

        self.open_ear_samples.append(ear)
        if len(self.open_ear_samples) < 8:
            return

        open_ear = sum(self.open_ear_samples) / len(self.open_ear_samples)
        self.closed_threshold = max(0.12, open_ear * 0.82)
        self.open_threshold = max(0.15, open_ear * 0.92)

    @staticmethod
    def _estimate_heart_rate(blink_rate):
        estimated_bpm = 60 + (blink_rate * 1.5)
        return int(round(min(110, max(60, estimated_bpm))))

    def reset(self):
        self.is_closed = False
        self.closed_frames = 0
        self.blink_count = 0
        self.blink_times.clear()
        self.open_ear_samples.clear()
        self.session_start = time.time()
        self.lowest_ear = 1.0

    def close(self):
        self.face_mesh.close()


class HeartBeatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Heart Beat Measurement Using Pupil")
        self.geometry("1280x800")
        self.minsize(1050, 680)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.cap = None
        self.running = False
        self.detector = BlinkDetector()
        self.photo = None
        self.camera_source = None
        self.metric_history = deque(maxlen=180)
        self.graph_canvases = {}
        self.last_graph_blink_count = 0

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.configure(fg_color="#0b1721")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, fg_color="#0b1721", corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=0)

        self.video_label = ctk.CTkLabel(left, text="", fg_color="#111f2b", corner_radius=6)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        graphs = self._panel(left)
        graphs.grid(row=1, column=0, sticky="ew", pady=(18, 12))
        graphs.grid_columnconfigure((0, 1), weight=1, uniform="graphs")
        ctk.CTkLabel(
            graphs,
            text="LIVE OUTPUT GRAPHS",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#1e9bff",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 8))
        self._add_graph(graphs, "blink_signal", "Blink Signal", "#ff1010", "Blink", 0, 1, row=1, column=0)
        self._add_graph(graphs, "heart_rate", "Heart Rate Graph", "#ffe600", "BPM", 50, 120, row=1, column=1)

        self.status_box = ctk.CTkFrame(left, fg_color="#101e2a", border_color="#253747", border_width=1)
        self.status_box.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(
            self.status_box,
            text="STATUS",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#1e9bff",
        ).pack(anchor="w", padx=18, pady=(14, 6))
        self.status_label = ctk.CTkLabel(
            self.status_box,
            text="Click START to begin blink detection.",
            font=ctk.CTkFont(size=16),
            text_color="#dce7ef",
        )
        self.status_label.pack(anchor="w", padx=18, pady=(0, 18))

        ctk.CTkLabel(
            left,
            text="Made with OpenCV and MediaPipe",
            font=ctk.CTkFont(size=15),
            text_color="#96a7b5",
        ).grid(row=3, column=0, sticky="w")

        right = ctk.CTkFrame(self, fg_color="#0b1721", corner_radius=0, width=370)
        right.grid(row=0, column=1, sticky="ns", padx=(8, 18), pady=18)
        right.grid_propagate(False)

        controls = self._panel(right)
        controls.pack(fill="x", pady=(0, 18))
        ctk.CTkLabel(
            controls,
            text="BLINK DETECTION (Both Eyes)",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#1e9bff",
        ).pack(anchor="w", padx=18, pady=(20, 16))
        buttons = ctk.CTkFrame(controls, fg_color="transparent")
        buttons.pack(fill="x", padx=18, pady=(0, 18))
        buttons.grid_columnconfigure((0, 1), weight=1)
        self.start_btn = ctk.CTkButton(
            buttons,
            text="START",
            height=50,
            fg_color="#268d2d",
            hover_color="#1f7a26",
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.start_detection,
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_btn = ctk.CTkButton(
            buttons,
            text="STOP",
            height=50,
            fg_color="#d92f2f",
            hover_color="#b92727",
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled",
            command=self.stop_detection,
        )
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        metrics = self._panel(right)
        metrics.pack(fill="x", pady=(0, 18))
        ctk.CTkLabel(
            metrics,
            text="REAL-TIME METRICS",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#1e9bff",
        ).pack(anchor="w", padx=18, pady=(20, 12))

        table = ctk.CTkFrame(metrics, fg_color="#0d1a25", border_color="#263747", border_width=1)
        table.pack(fill="x", padx=18, pady=(0, 18))
        self.metric_labels = {}
        rows = [
            ("Blink Count", "blink_count", "#1e9bff"),
            ("Heart Rate", "heart_rate", "#ff3535"),
            ("Frequency", "frequency", "#32ff32"),
            ("Confidence", "confidence", "#ffe928"),
            ("Status", "status", "#32ff32"),
        ]
        for row, (label, key, color) in enumerate(rows):
            table.grid_columnconfigure(0, weight=1)
            table.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(table, text=label, font=ctk.CTkFont(size=15), anchor="w").grid(
                row=row, column=0, sticky="ew", padx=14, pady=12
            )
            value = ctk.CTkLabel(
                table,
                text="0",
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=color,
                anchor="center",
            )
            value.grid(row=row, column=1, sticky="ew", padx=14, pady=12)
            self.metric_labels[key] = value

        instructions = self._panel(right)
        instructions.pack(fill="both", expand=True)
        ctk.CTkLabel(
            instructions,
            text="INSTRUCTIONS",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#1e9bff",
        ).pack(anchor="w", padx=18, pady=(20, 12))
        for text in [
            "Click START to begin blink detection",
            "Look at the camera and blink naturally",
            "Ensure good lighting on your face",
            "Keep your head steady",
            "Click STOP to end detection",
        ]:
            ctk.CTkLabel(
                instructions,
                text=f"-  {text}",
                font=ctk.CTkFont(size=15),
                text_color="#dce7ef",
            ).pack(anchor="w", padx=22, pady=5)

    @staticmethod
    def _panel(parent):
        return ctk.CTkFrame(parent, fg_color="#101e2a", border_color="#253747", border_width=1)

    def _add_graph(self, parent, key, title, color, unit, min_value, max_value, row=None, column=None):
        wrapper = ctk.CTkFrame(parent, fg_color="transparent")
        if row is None or column is None:
            wrapper.pack(fill="x", padx=18, pady=(0, 14))
        else:
            wrapper.grid(row=row, column=column, sticky="ew", padx=(18, 9) if column == 0 else (9, 18), pady=(0, 18))
        ctk.CTkLabel(
            wrapper,
            text=f"{title} ({unit})",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#dce7ef",
        ).pack(anchor="w", pady=(0, 6))
        canvas = tk.Canvas(
            wrapper,
            height=145,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#4a4a4a",
        )
        canvas.pack(fill="x")
        self.graph_canvases[key] = {
            "canvas": canvas,
            "title": title,
            "color": color,
            "unit": unit,
            "min": min_value,
            "max": max_value,
        }
        canvas.bind("<Configure>", lambda _event, graph_key=key: self._draw_graph(graph_key))

    def start_detection(self):
        self.cap, self.camera_source = self._open_camera()
        if self.cap is None:
            self.status_label.configure(text="Camera not found. Check camera permission or device.")
            self.metric_labels["status"].configure(text="No Camera", text_color="#ff3535")
            self.cap = None
            return

        self.detector.reset()
        self.metric_history.clear()
        self.last_graph_blink_count = 0
        self._draw_all_graphs()
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text=f"Blink detection is running on camera {self.camera_source}...")
        self.metric_labels["status"].configure(text="Running", text_color="#32ff32")
        self.update_frame()

    @staticmethod
    def _open_camera():
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for index in range(3):
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    return cap, index
                cap.release()
        return None, None

    def stop_detection(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_source = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Blink detection stopped.")
        self.metric_labels["status"].configure(text="Stopped", text_color="#ff3535")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_label.configure(text="Failed to read frame from camera.")
            self._draw_placeholder("Failed to read frame from camera.")
            self.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        metrics = self.detector.process(frame)
        self._draw_overlay(frame, metrics)
        self._update_metrics(metrics)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_w = max(self.video_label.winfo_width(), 800)
        label_h = max(self.video_label.winfo_height(), 520)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

        self.after(15, self.update_frame)

    def _draw_placeholder(self, message):
        width = max(self.video_label.winfo_width(), 800)
        height = max(self.video_label.winfo_height(), 520)
        image = Image.new("RGB", (width, height), "#111f2b")
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text=message, text_color="#dce7ef")

    def _draw_overlay(self, frame, metrics):
        if not metrics["face_detected"]:
            cv2.putText(
                frame,
                "No face detected",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Look at the camera in good light",
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            return

        color_map = [
            (f"Blink Count : {metrics['blink_count']}", (255, 140, 0)),
            (f"Heart Rate : {metrics['heart_rate']} BPM", (45, 45, 255)),
            (f"Frequency : {metrics['frequency']:.2f} Hz", (40, 255, 40)),
            (f"Confidence : {metrics['confidence']:.2f} % ({self._confidence_text(metrics['confidence'])})", (0, 255, 255)),
        ]
        y = 40
        for text, color in color_map:
            cv2.putText(frame, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += 38

    def _update_metrics(self, metrics):
        self.metric_labels["blink_count"].configure(text=str(metrics["blink_count"]))
        self.metric_labels["heart_rate"].configure(text=f"{metrics['heart_rate']} BPM")
        self.metric_labels["frequency"].configure(text=f"{metrics['frequency']:.2f} Hz")
        self.metric_labels["confidence"].configure(
            text=f"{metrics['confidence']:.2f} % ({self._confidence_text(metrics['confidence'])})"
        )
        self._record_graph_sample(metrics)
        if metrics["face_detected"]:
            self.status_label.configure(text=f"Blink detection is running on camera {self.camera_source}...")
            self.metric_labels["status"].configure(text="Face Detected", text_color="#32ff32")
        else:
            self.status_label.configure(text="No face detected. Look at the camera.")
            self.metric_labels["status"].configure(text="No Face", text_color="#ffe928")

    def _record_graph_sample(self, metrics):
        if metrics["face_detected"]:
            blink_signal = 1 if metrics["blink_count"] > self.last_graph_blink_count else 0
            self.last_graph_blink_count = metrics["blink_count"]
            sample = {
                "heart_rate": metrics["heart_rate"],
                "blink_signal": blink_signal,
            }
        else:
            sample = {
                "heart_rate": None,
                "blink_signal": None,
            }
        self.metric_history.append(sample)
        self._draw_all_graphs()

    def _draw_all_graphs(self):
        for key in self.graph_canvases:
            self._draw_graph(key)

    def _draw_graph(self, key):
        graph = self.graph_canvases.get(key)
        if graph is None:
            return

        canvas = graph["canvas"]
        canvas.delete("all")

        width = max(canvas.winfo_width(), 260)
        height = max(canvas.winfo_height(), 105)
        pad_left, pad_right, pad_top, pad_bottom = 38, 10, 12, 24
        plot_w = max(width - pad_left - pad_right, 1)
        plot_h = max(height - pad_top - pad_bottom, 1)
        min_value = graph["min"]
        max_value = graph["max"]
        color = graph["color"]

        canvas.create_rectangle(0, 0, width, height, fill="#ffffff", outline="")
        for fraction in (0.0, 0.5, 1.0):
            y = pad_top + (1.0 - fraction) * plot_h
            value = min_value + fraction * (max_value - min_value)
            canvas.create_line(pad_left, y, width - pad_right, y, fill="#dddddd", width=1)
            canvas.create_text(
                8,
                y,
                text=f"{value:.1f}" if max_value - min_value <= 2 else f"{value:.0f}",
                fill="#333333",
                anchor="w",
                font=("Segoe UI", 8),
            )
        canvas.create_line(pad_left, pad_top, pad_left, height - pad_bottom, fill="#333333", width=1)
        canvas.create_line(pad_left, height - pad_bottom, width - pad_right, height - pad_bottom, fill="#333333", width=1)

        values = [sample[key] for sample in self.metric_history if sample[key] is not None]
        if not values:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for face detection",
                fill="#555555",
                font=("Segoe UI", 10),
            )
            return

        visible = values[-90:]
        points = []
        denom = max(len(visible) - 1, 1)
        for index, value in enumerate(visible):
            clamped = min(max_value, max(min_value, value))
            x = pad_left + (index / denom) * plot_w
            y = pad_top + (1.0 - ((clamped - min_value) / (max_value - min_value))) * plot_h
            points.extend((x, y))

        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=3, smooth=(key != "blink_signal"))
        else:
            x, y = points
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

        latest = visible[-1]
        canvas.create_text(
            width - pad_right,
            height - 10,
            text=f"{latest:.1f} {graph['unit']}",
            fill=color,
            anchor="e",
            font=("Segoe UI", 10, "bold"),
        )

    @staticmethod
    def _confidence_text(confidence):
        if confidence >= 80:
            return "High"
        if confidence >= 60:
            return "Medium"
        return "Low"

    def on_close(self):
        self.stop_detection()
        self.detector.close()
        self.destroy()


if __name__ == "__main__":
    app = HeartBeatApp()
    app.mainloop()
