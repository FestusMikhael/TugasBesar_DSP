# rppg.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import threading
import time

# --- Inisialisasi model face detection ---
base_model = "models/blaze_face_short_range.tflite"
base_options = python.BaseOptions(model_asset_path=base_model)
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

options = FaceDetectorOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
)
face_detector = vision.FaceDetector.create_from_options(options)

# --- Parameter bounding box ---
margin_x = 10
scaling_factor = 0.8

def rppg_process(rgb_frame, frame):
    """
    Memproses satu frame untuk mendeteksi wajah dan mengambil rata-rata RGB dari ROI.
    Mengembalikan nilai RGB rata-rata dan frame dengan bbox (jika terdeteksi).
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = face_detector.detect(mp_image)
    if result.detections:
        for detection in result.detections:
            bboxC = detection.bounding_box
            x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height

            new_x = int(x + margin_x)
            new_w = int(w * scaling_factor)
            new_h = int(h * scaling_factor)

            face_roi = rgb_frame[y:y+new_h, new_x:new_x+new_w]

            if face_roi.size == 0:
                return None, frame

            mean_rgb = cv2.mean(face_roi)[:3]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + new_w), int(y + new_h)), (0, 255, 0), 2)
            return mean_rgb, frame

    return None, frame

# --- Bagian tambahan: Kamera dan Grafik ---
def live_plot(rgb_buffer, interval=0.05):
    plt.ion()
    fig, ax = plt.subplots()
    line_r, = ax.plot([], [], label='R', color='red')
    line_g, = ax.plot([], [], label='G', color='green')
    line_b, = ax.plot([], [], label='B', color='blue')
    ax.set_ylim(0, 255)
    ax.set_xlim(0, 100)
    ax.legend()
    
    while True:
        if len(rgb_buffer) > 0:
            data = np.array(rgb_buffer)
            line_r.set_ydata(data[:, 0])
            line_g.set_ydata(data[:, 1])
            line_b.set_ydata(data[:, 2])
            line_r.set_xdata(np.arange(len(data)))
            line_g.set_xdata(np.arange(len(data)))
            line_b.set_xdata(np.arange(len(data)))
            ax.set_xlim(0, len(data))
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(interval)

def main():
    cap = cv2.VideoCapture(0)
    rgb_values = deque(maxlen=100)

    # Jalankan plot di thread terpisah
    plot_thread = threading.Thread(target=live_plot, args=(rgb_values,))
    plot_thread.daemon = True
    plot_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mean_rgb, frame_with_box = rppg_process(rgb_frame, frame)

        if mean_rgb:
            rgb_values.append(mean_rgb)

        cv2.imshow("rPPG Frame", frame_with_box)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Jalankan jika file ini langsung dijalankan ---
if __name__ == "__main__":
    main()
