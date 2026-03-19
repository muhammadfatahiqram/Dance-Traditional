import csv
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import font, filedialog
import threading
from PIL import Image, ImageTk
import time
from collections import defaultdict

last_label = "Nama_Gerakan_Tari"

# Untuk FPS
prev_time = 0

# Untuk confidence durasi label
label_start_time = time.time()
label_duration = 0

# Untuk rekap CSV otomatis
gesture_counter = defaultdict(int)

last_label = "Nama_Gerakan_Tari"

def show_start_window():
    root = tk.Tk()
    root.title("Sistem Deteksi Pose")
    root.geometry("1280x720")
    root.configure(bg='black')

    # Font besar
    big_font = font.Font(family='Helvetica', size=36, weight='bold')

    label = tk.Label(root, text="Sistem Deteksi Pose Tari", font=big_font, fg='white', bg='black')
    label.pack(pady=100)

    def start_program():
        root.destroy()  # Tutup GUI
        threading.Thread(target=main).start()  # Jalankan proses utama

    start_button = tk.Button(root, text="TEKAN UNTUK MULAI", font=big_font, bg='green', fg='white',
                             padx=20, pady=10, command=start_program)
    start_button.pack(pady=50)

    root.mainloop()

# Inisialisasi pose Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Deteksi pose dari gambar
def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image,
                                  landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width),
                              int(landmark.y * height),
                              landmark.z * width))

    if display:
        plt.figure(figsize=[22, 22])
        plt.subplot(121); plt.imshow(image[:, :, ::-1]); plt.title("Original Image"); plt.axis('off')
        plt.subplot(122); plt.imshow(output_image[:, :, ::-1]); plt.title("Output Image"); plt.axis('off')
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks

# Hitung sudut dari 3 landmark
def calculateAngle(p1, p2, p3):
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                         math.atan2(y1 - y2, x1 - x2))
    return angle + 360 if angle < 0 else angle

# Klasifikasi pose berdasarkan sudut
def classifyPose(landmarks, output_image, display=False):
    global last_label, label_start_time, label_duration

    label = last_label
    color = (0, 0, 255)

    left_elbow_angle = calculateAngle(landmarks[11], landmarks[13], landmarks[15])
    right_elbow_angle = calculateAngle(landmarks[12], landmarks[14], landmarks[16])
    left_shoulder_angle = calculateAngle(landmarks[13], landmarks[11], landmarks[23])
    right_shoulder_angle = calculateAngle(landmarks[24], landmarks[12], landmarks[14])
    left_knee_angle = calculateAngle(landmarks[23], landmarks[25], landmarks[27])
    right_knee_angle = calculateAngle(landmarks[24], landmarks[26], landmarks[28])

    # ================================
    # RULE BASED CLASSIFICATION
    # ================================
    if left_elbow_angle > 150 and left_elbow_angle < 165 and right_elbow_angle > 40 and right_elbow_angle < 70 or \
        left_elbow_angle > 140 and left_elbow_angle < 170 and right_elbow_angle > 25 and right_elbow_angle < 50 or \
        left_elbow_angle > 140 and left_elbow_angle < 150 and right_elbow_angle > 30 and right_elbow_angle < 40 or \
        left_elbow_angle > 155 and left_elbow_angle < 165 and right_elbow_angle > 30 and right_elbow_angle < 40 or \
        left_elbow_angle > 155 and left_elbow_angle < 165 and right_elbow_angle > 55 and right_elbow_angle < 70 or \
        left_elbow_angle > 155 and left_elbow_angle < 165 and right_elbow_angle > 5 and right_elbow_angle < 20 or \
        left_elbow_angle > 285 and left_elbow_angle < 295 and right_elbow_angle > 190 and right_elbow_angle < 200 or \
        left_elbow_angle > 154 and left_elbow_angle < 158 and right_elbow_angle > 36  and right_elbow_angle < 40  or \
        left_elbow_angle > 160 and left_elbow_angle < 164 and right_elbow_angle > 32  and right_elbow_angle < 36  or \
        left_elbow_angle > 161 and left_elbow_angle < 165 and right_elbow_angle > 34  and right_elbow_angle < 38  or \
        left_elbow_angle > 160 and left_elbow_angle < 164 and right_elbow_angle > 35  and right_elbow_angle < 39  or \
        left_elbow_angle > 160 and left_elbow_angle < 164 and right_elbow_angle > 36  and right_elbow_angle < 40  or \
        left_elbow_angle > 165 and left_elbow_angle < 169 and right_elbow_angle > 37  and right_elbow_angle < 41  or \
        left_elbow_angle > 213 and left_elbow_angle < 217 and right_elbow_angle > 185 and right_elbow_angle < 189 or \
        left_elbow_angle > 272 and left_elbow_angle < 276 and right_elbow_angle > 188 and right_elbow_angle < 192 or \
        left_elbow_angle > 298 and left_elbow_angle < 302 and right_elbow_angle > 188 and right_elbow_angle < 192 or \
        left_elbow_angle > 305 and left_elbow_angle < 309 and right_elbow_angle > 194 and right_elbow_angle < 198 or \
        left_elbow_angle > 297 and left_elbow_angle < 301 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 287 and left_elbow_angle < 291 and right_elbow_angle > 180 and right_elbow_angle < 184 or \
        left_elbow_angle > 291 and left_elbow_angle < 295 and right_elbow_angle > 182 and right_elbow_angle < 186 or \
        left_elbow_angle > 295 and left_elbow_angle < 299 and right_elbow_angle > 181 and right_elbow_angle < 185 or \
        left_elbow_angle > 293 and left_elbow_angle < 297 and right_elbow_angle > 185 and right_elbow_angle < 189 or \
        left_elbow_angle > 277 and left_elbow_angle < 281 and right_elbow_angle > 182 and right_elbow_angle < 186 or \
        left_elbow_angle > 223 and left_elbow_angle < 227 and right_elbow_angle > 180 and right_elbow_angle < 184 or \
        left_elbow_angle > 154 and left_elbow_angle < 158 and right_elbow_angle > 73  and right_elbow_angle < 77  or \
        left_elbow_angle > 149 and left_elbow_angle < 153 and right_elbow_angle > 35  and right_elbow_angle < 39  or \
        left_elbow_angle > 146 and left_elbow_angle < 150 and right_elbow_angle > 25  and right_elbow_angle < 29  or \
        left_elbow_angle > 146 and left_elbow_angle < 150 and right_elbow_angle > 28  and right_elbow_angle < 32  or \
        left_elbow_angle > 152 and left_elbow_angle < 156 and right_elbow_angle > 37  and right_elbow_angle < 41  or \
        left_elbow_angle > 154 and left_elbow_angle < 158 and right_elbow_angle > 36  and right_elbow_angle < 40  or \
        left_elbow_angle > 153 and left_elbow_angle < 157 and right_elbow_angle > 33  and right_elbow_angle < 37  or \
        left_elbow_angle > 266 and left_elbow_angle < 270 and right_elbow_angle > 179 and right_elbow_angle < 183 or \
        left_elbow_angle > 287 and left_elbow_angle < 291 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 303 and left_elbow_angle < 307 and right_elbow_angle > 186 and right_elbow_angle < 190 or \
        left_elbow_angle > 313 and left_elbow_angle < 317 and right_elbow_angle > 200 and right_elbow_angle < 204 or \
        left_elbow_angle > 309 and left_elbow_angle < 313 and right_elbow_angle > 202 and right_elbow_angle < 206 or \
        left_elbow_angle > 288 and left_elbow_angle < 292 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 284 and left_elbow_angle < 288 and right_elbow_angle > 182 and right_elbow_angle < 186 or \
        left_elbow_angle > 279 and left_elbow_angle < 283 and right_elbow_angle > 185 and right_elbow_angle < 189 or \
        left_elbow_angle > 139 and left_elbow_angle < 143 and right_elbow_angle > 56  and right_elbow_angle < 60  or \
        left_elbow_angle > 142 and left_elbow_angle < 146 and right_elbow_angle > 52  and right_elbow_angle < 56  or \
        left_elbow_angle > 154 and left_elbow_angle < 158 and right_elbow_angle > 68  and right_elbow_angle < 72  or \
        left_elbow_angle > 156 and left_elbow_angle < 160 and right_elbow_angle > 80  and right_elbow_angle < 84  or \
        left_elbow_angle > 149 and left_elbow_angle < 153 and right_elbow_angle > 75  and right_elbow_angle < 79  or \
        left_elbow_angle > 239 and left_elbow_angle < 243 and right_elbow_angle > 196 and right_elbow_angle < 200 or \
        left_elbow_angle > 300 and left_elbow_angle < 304 and right_elbow_angle > 199 and right_elbow_angle < 203 or \
        left_elbow_angle > 303 and left_elbow_angle < 307 and right_elbow_angle > 199 and right_elbow_angle < 203 or \
        left_elbow_angle > 300 and left_elbow_angle < 304 and right_elbow_angle > 203 and right_elbow_angle < 207 or \
        left_elbow_angle > 295 and left_elbow_angle < 299 and right_elbow_angle > 203 and right_elbow_angle < 207 or \
        left_elbow_angle > 294 and left_elbow_angle < 298 and right_elbow_angle > 201 and right_elbow_angle < 205 or \
        left_elbow_angle > 288 and left_elbow_angle < 292 and right_elbow_angle > 201 and right_elbow_angle < 205 or \
        left_elbow_angle > 233 and left_elbow_angle < 237 and right_elbow_angle > 197 and right_elbow_angle < 201 or \
        left_elbow_angle > 198 and left_elbow_angle < 202 and right_elbow_angle > 186 and right_elbow_angle < 190 or \
        left_elbow_angle > 137 and left_elbow_angle < 141 and right_elbow_angle > 62  and right_elbow_angle < 66  or \
        left_elbow_angle > 136 and left_elbow_angle < 140 and right_elbow_angle > 56  and right_elbow_angle < 60  or \
        left_elbow_angle > 151 and left_elbow_angle < 155 and right_elbow_angle > 57  and right_elbow_angle < 61  or \
        left_elbow_angle > 154 and left_elbow_angle < 158 and right_elbow_angle > 70  and right_elbow_angle < 74  or \
        left_elbow_angle > 151 and left_elbow_angle < 155 and right_elbow_angle > 68  and right_elbow_angle < 72  or \
        left_elbow_angle > 154 and left_elbow_angle < 158 and right_elbow_angle > 66  and right_elbow_angle < 70  or \
        left_elbow_angle > 148 and left_elbow_angle < 152 and right_elbow_angle > 77  and right_elbow_angle < 81 or \
        left_elbow_angle > 150 and left_elbow_angle < 154 and right_elbow_angle > 109 and right_elbow_angle < 113 or \
        left_elbow_angle > 256 and left_elbow_angle < 260 and right_elbow_angle > 199 and right_elbow_angle < 203 or \
        left_elbow_angle > 298 and left_elbow_angle < 302 and right_elbow_angle > 211 and right_elbow_angle < 215 or \
        left_elbow_angle > 310 and left_elbow_angle < 314 and right_elbow_angle > 193 and right_elbow_angle < 197 or \
        left_elbow_angle > 305 and left_elbow_angle < 309 and right_elbow_angle > 196 and right_elbow_angle < 200 or \
        left_elbow_angle > 304 and left_elbow_angle < 308 and right_elbow_angle > 196 and right_elbow_angle < 200 or \
        left_elbow_angle > 307 and left_elbow_angle < 311 and right_elbow_angle > 198 and right_elbow_angle < 202 or \
        left_elbow_angle > 307 and left_elbow_angle < 311 and right_elbow_angle > 208 and right_elbow_angle < 212 or \
        left_elbow_angle > 306 and left_elbow_angle < 310 and right_elbow_angle > 205 and right_elbow_angle < 209 or \
        left_elbow_angle > 302 and left_elbow_angle < 306 and right_elbow_angle > 203 and right_elbow_angle < 207 or \
        left_elbow_angle > 301 and left_elbow_angle < 305 and right_elbow_angle > 207 and right_elbow_angle < 211 or \
        left_elbow_angle > 296 and left_elbow_angle < 300 and right_elbow_angle > 211 and right_elbow_angle < 215 or \
        left_elbow_angle > 236 and left_elbow_angle < 240 and right_elbow_angle > 190 and right_elbow_angle < 194 or \
        left_elbow_angle > 173 and left_elbow_angle < 177 and right_elbow_angle > 85  and right_elbow_angle < 89  or \
        left_elbow_angle > 168 and left_elbow_angle < 172 and right_elbow_angle > 37  and right_elbow_angle < 41  or \
        left_elbow_angle > 167 and left_elbow_angle < 171 and right_elbow_angle > 23  and right_elbow_angle < 27  or \
        left_elbow_angle > 166 and left_elbow_angle < 170 and right_elbow_angle > 0   and right_elbow_angle < 4   or \
        left_elbow_angle > 172 and left_elbow_angle < 176 and right_elbow_angle > 19  and right_elbow_angle < 23  or \
        left_elbow_angle > 173 and left_elbow_angle < 177 and right_elbow_angle > 30  and right_elbow_angle < 34  or \
        left_elbow_angle > 171 and left_elbow_angle < 175 and right_elbow_angle > 36  and right_elbow_angle < 40  or \
        left_elbow_angle > 167 and left_elbow_angle < 171 and right_elbow_angle > 31  and right_elbow_angle < 35  or \
        left_elbow_angle > 229 and left_elbow_angle < 233 and right_elbow_angle > 184 and right_elbow_angle < 188 or \
        left_elbow_angle > 276 and left_elbow_angle < 280 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 332 and left_elbow_angle < 336 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 342 and left_elbow_angle < 346 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 347 and left_elbow_angle < 351 and right_elbow_angle > 185 and right_elbow_angle < 189 or \
        left_elbow_angle > 328 and left_elbow_angle < 332 and right_elbow_angle > 179 and right_elbow_angle < 183 or \
        left_elbow_angle > 321 and left_elbow_angle < 325 and right_elbow_angle > 172 and right_elbow_angle < 176 or \
        left_elbow_angle > 330 and left_elbow_angle < 334 and right_elbow_angle > 178 and right_elbow_angle < 182 or \
        left_elbow_angle > 186 and left_elbow_angle < 190 and right_elbow_angle > 174 and right_elbow_angle < 178 or \
        left_elbow_angle > 171 and left_elbow_angle < 175 and right_elbow_angle > 51  and right_elbow_angle < 55  or \
        left_elbow_angle > 162 and left_elbow_angle < 166 and right_elbow_angle > 18  and right_elbow_angle < 22  or \
        left_elbow_angle > 158 and left_elbow_angle < 162 and right_elbow_angle > 10  and right_elbow_angle < 14  or \
        left_elbow_angle > 167 and left_elbow_angle < 171 and right_elbow_angle > 42  and right_elbow_angle < 46  or \
        left_elbow_angle > 167 and left_elbow_angle < 171 and right_elbow_angle > 44  and right_elbow_angle < 48  or \
        left_elbow_angle > 165 and left_elbow_angle < 169 and right_elbow_angle > 43  and right_elbow_angle < 47  or \
        left_elbow_angle > 170 and left_elbow_angle < 174 and right_elbow_angle > 127 and right_elbow_angle < 131 or \
        left_elbow_angle > 246 and left_elbow_angle < 250 and right_elbow_angle > 186 and right_elbow_angle < 190 or \
        left_elbow_angle > 329 and left_elbow_angle < 333 and right_elbow_angle > 181 and right_elbow_angle < 185 or \
        left_elbow_angle > 336 and left_elbow_angle < 340 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 346 and left_elbow_angle < 350 and right_elbow_angle > 182 and right_elbow_angle < 186 or \
        left_elbow_angle > 343 and left_elbow_angle < 347 and right_elbow_angle > 177 and right_elbow_angle < 181 or \
        left_elbow_angle > 324 and left_elbow_angle < 328 and right_elbow_angle > 171 and right_elbow_angle < 175 or \
        left_elbow_angle > 344 and left_elbow_angle < 348 and right_elbow_angle > 178 and right_elbow_angle < 182 or \
        left_elbow_angle > 331 and left_elbow_angle < 335 and right_elbow_angle > 175 and right_elbow_angle < 179 or \
        left_elbow_angle > 339 and left_elbow_angle < 343 and right_elbow_angle > 176 and right_elbow_angle < 180 or \
        left_elbow_angle > 330 and left_elbow_angle < 334 and right_elbow_angle > 177 and right_elbow_angle < 181 or \
        left_elbow_angle > 331 and left_elbow_angle < 335 and right_elbow_angle > 175 and right_elbow_angle < 179 or \
        left_elbow_angle > 290 and left_elbow_angle < 310 and right_elbow_angle > 195 and right_elbow_angle < 210:
        if left_shoulder_angle > 20 and left_shoulder_angle < 35 and right_shoulder_angle > 40 and right_shoulder_angle < 50 or \
            left_shoulder_angle > 35 and left_shoulder_angle < 45 and right_shoulder_angle > 30 and right_shoulder_angle < 55 or\
            left_shoulder_angle > 30 and left_shoulder_angle < 45 and right_shoulder_angle > 65 and right_shoulder_angle < 80 or\
            left_shoulder_angle > 40 and left_shoulder_angle < 55 and right_shoulder_angle > 45 and right_shoulder_angle < 55 or\
            left_shoulder_angle > 71  and left_shoulder_angle < 75  and right_shoulder_angle > 39  and right_shoulder_angle < 43  or \
            left_shoulder_angle > 64  and left_shoulder_angle < 68  and right_shoulder_angle > 46  and right_shoulder_angle < 50  or \
            left_shoulder_angle > 58  and left_shoulder_angle < 62  and right_shoulder_angle > 42  and right_shoulder_angle < 46  or \
            left_shoulder_angle > 50  and left_shoulder_angle < 54  and right_shoulder_angle > 41  and right_shoulder_angle < 45  or \
            left_shoulder_angle > 49  and left_shoulder_angle < 53  and right_shoulder_angle > 41  and right_shoulder_angle < 45  or \
            left_shoulder_angle > 47  and left_shoulder_angle < 51  and right_shoulder_angle > 43  and right_shoulder_angle < 47  or \
            left_shoulder_angle > 2   and left_shoulder_angle < 6   and right_shoulder_angle > 44  and right_shoulder_angle < 48  or \
            left_shoulder_angle > 9   and left_shoulder_angle < 13  and right_shoulder_angle > 62  and right_shoulder_angle < 66  or \
            left_shoulder_angle > 17  and left_shoulder_angle < 21  and right_shoulder_angle > 74  and right_shoulder_angle < 78  or \
            left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 82  and right_shoulder_angle < 86  or \
            left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 87  and right_shoulder_angle < 91  or \
            left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 75  and right_shoulder_angle < 79  or \
            left_shoulder_angle > 24  and left_shoulder_angle < 28  and right_shoulder_angle > 89  and right_shoulder_angle < 73  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 70  and right_shoulder_angle < 74  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 68  and right_shoulder_angle < 72  or \
            left_shoulder_angle > 24  and left_shoulder_angle < 28  and right_shoulder_angle > 67  and right_shoulder_angle < 71  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 51  and right_shoulder_angle < 55  or \
            left_shoulder_angle > 54  and left_shoulder_angle < 58  and right_shoulder_angle > 30  and right_shoulder_angle < 34  or \
            left_shoulder_angle > 58  and left_shoulder_angle < 62  and right_shoulder_angle > 38  and right_shoulder_angle < 42  or \
            left_shoulder_angle > 61  and left_shoulder_angle < 65  and right_shoulder_angle > 44  and right_shoulder_angle < 48  or \
            left_shoulder_angle > 52  and left_shoulder_angle < 56  and right_shoulder_angle > 40  and right_shoulder_angle < 44  or \
            left_shoulder_angle > 43  and left_shoulder_angle < 47  and right_shoulder_angle > 36  and right_shoulder_angle < 40  or \
            left_shoulder_angle > 39  and left_shoulder_angle < 43  and right_shoulder_angle > 37  and right_shoulder_angle < 41  or \
            left_shoulder_angle > 42  and left_shoulder_angle < 46  and right_shoulder_angle > 39  and right_shoulder_angle < 43  or \
            left_shoulder_angle > 13  and left_shoulder_angle < 17  and right_shoulder_angle > 53  and right_shoulder_angle < 57  or \
            left_shoulder_angle > 16  and left_shoulder_angle < 20  and right_shoulder_angle > 67  and right_shoulder_angle < 71  or \
            left_shoulder_angle > 17  and left_shoulder_angle < 21  and right_shoulder_angle > 82  and right_shoulder_angle < 86  or \
            left_shoulder_angle > 21  and left_shoulder_angle < 85  and right_shoulder_angle > 88  and right_shoulder_angle < 92  or \
            left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 83  and right_shoulder_angle < 87  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 79  and right_shoulder_angle < 83  or \
            left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 70  and right_shoulder_angle < 74  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 62  and right_shoulder_angle < 66  or \
            left_shoulder_angle > 60  and left_shoulder_angle < 62  and right_shoulder_angle > 31  and right_shoulder_angle < 35  or \
            left_shoulder_angle > 76  and left_shoulder_angle < 80  and right_shoulder_angle > 30  and right_shoulder_angle < 34  or \
            left_shoulder_angle > 89  and left_shoulder_angle < 93  and right_shoulder_angle > 19  and right_shoulder_angle < 23  or \
            left_shoulder_angle > 77  and left_shoulder_angle < 81  and right_shoulder_angle > 9   and right_shoulder_angle < 13  or \
            left_shoulder_angle > 75  and left_shoulder_angle < 79  and right_shoulder_angle > 16  and right_shoulder_angle < 20  or \
            left_shoulder_angle > 29  and left_shoulder_angle < 33  and right_shoulder_angle > 56  and right_shoulder_angle < 60  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 71  and right_shoulder_angle < 75  or \
            left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 77  and right_shoulder_angle < 81  or \
            left_shoulder_angle > 10  and left_shoulder_angle < 14  and right_shoulder_angle > 81  and right_shoulder_angle < 85  or \
            left_shoulder_angle > 8   and left_shoulder_angle < 12  and right_shoulder_angle > 79  and right_shoulder_angle < 83  or \
            left_shoulder_angle > 10  and left_shoulder_angle < 14  and right_shoulder_angle > 80  and right_shoulder_angle < 84  or \
            left_shoulder_angle > 17  and left_shoulder_angle < 21  and right_shoulder_angle > 75  and right_shoulder_angle < 79  or \
            left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 55  and right_shoulder_angle < 59  or \
            left_shoulder_angle > 20  and left_shoulder_angle < 24  and right_shoulder_angle > 42  and right_shoulder_angle < 46  or \
            left_shoulder_angle > 54  and left_shoulder_angle < 58  and right_shoulder_angle > 28  and right_shoulder_angle < 32  or \
            left_shoulder_angle > 69  and left_shoulder_angle < 73  and right_shoulder_angle > 30  and right_shoulder_angle < 34  or \
            left_shoulder_angle > 86  and left_shoulder_angle < 90  and right_shoulder_angle > 28  and right_shoulder_angle < 32  or \
            left_shoulder_angle > 81  and left_shoulder_angle < 85  and right_shoulder_angle > 19  and right_shoulder_angle < 23  or \
            left_shoulder_angle > 75  and left_shoulder_angle < 79  and right_shoulder_angle > 12  and right_shoulder_angle < 16  or \
            left_shoulder_angle > 69  and left_shoulder_angle < 73  and right_shoulder_angle > 19  and right_shoulder_angle < 23  or \
            left_shoulder_angle > 58  and left_shoulder_angle < 62  and right_shoulder_angle > 13  and right_shoulder_angle < 17  or \
            left_shoulder_angle > 46  and left_shoulder_angle < 50  and right_shoulder_angle > 12  and right_shoulder_angle < 16  or \
            left_shoulder_angle > 21  and left_shoulder_angle < 25  and right_shoulder_angle > 49  and right_shoulder_angle < 53  or \
            left_shoulder_angle > 19  and left_shoulder_angle < 23  and right_shoulder_angle > 60  and right_shoulder_angle < 64  or \
            left_shoulder_angle > 16  and left_shoulder_angle < 20  and right_shoulder_angle > 85  and right_shoulder_angle < 89  or \
            left_shoulder_angle > 12  and left_shoulder_angle < 16  and right_shoulder_angle > 84  and right_shoulder_angle < 88  or \
            left_shoulder_angle > 15  and left_shoulder_angle < 19  and right_shoulder_angle > 82  and right_shoulder_angle < 86  or \
            left_shoulder_angle > 19  and left_shoulder_angle < 23  and right_shoulder_angle > 76  and right_shoulder_angle < 80  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 73  and right_shoulder_angle < 77  or \
            left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 75  and right_shoulder_angle < 79  or \
            left_shoulder_angle > 24  and left_shoulder_angle < 28  and right_shoulder_angle > 76  and right_shoulder_angle < 80  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 73  and right_shoulder_angle < 77  or \
            left_shoulder_angle > 21  and left_shoulder_angle < 25  and right_shoulder_angle > 69  and right_shoulder_angle < 73  or \
            left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 48  and right_shoulder_angle < 52  or \
            left_shoulder_angle > 50  and left_shoulder_angle < 54  and right_shoulder_angle > 42  and right_shoulder_angle < 46  or \
            left_shoulder_angle > 63  and left_shoulder_angle < 67  and right_shoulder_angle > 62  and right_shoulder_angle < 66  or \
            left_shoulder_angle > 70  and left_shoulder_angle < 74  and right_shoulder_angle > 74  and right_shoulder_angle < 78  or \
            left_shoulder_angle > 82  and left_shoulder_angle < 86  and right_shoulder_angle > 108 and right_shoulder_angle < 112 or \
            left_shoulder_angle > 83  and left_shoulder_angle < 87  and right_shoulder_angle > 68  and right_shoulder_angle < 72  or \
            left_shoulder_angle > 77  and left_shoulder_angle < 81  and right_shoulder_angle > 56  and right_shoulder_angle < 60  or \
            left_shoulder_angle > 73  and left_shoulder_angle < 77  and right_shoulder_angle > 52  and right_shoulder_angle < 56  or \
            left_shoulder_angle > 76  and left_shoulder_angle < 80  and right_shoulder_angle > 58  and right_shoulder_angle < 62  or \
            left_shoulder_angle > 10  and left_shoulder_angle < 14  and right_shoulder_angle > 26  and right_shoulder_angle < 30  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 45  and right_shoulder_angle < 49  or \
            left_shoulder_angle > 57  and left_shoulder_angle < 61  and right_shoulder_angle > 66  and right_shoulder_angle < 70  or \
            left_shoulder_angle > 66  and left_shoulder_angle < 70  and right_shoulder_angle > 81  and right_shoulder_angle < 85  or \
            left_shoulder_angle > 38  and left_shoulder_angle < 42  and right_shoulder_angle > 83  and right_shoulder_angle < 87  or \
            left_shoulder_angle > 31  and left_shoulder_angle < 35  and right_shoulder_angle > 77  and right_shoulder_angle < 81  or \
            left_shoulder_angle > 27  and left_shoulder_angle < 31  and right_shoulder_angle > 73  and right_shoulder_angle < 77  or \
            left_shoulder_angle > 31  and left_shoulder_angle < 35  and right_shoulder_angle > 78  and right_shoulder_angle < 82  or \
            left_shoulder_angle > 11  and left_shoulder_angle < 15  and right_shoulder_angle > 56  and right_shoulder_angle < 60  or \
            left_shoulder_angle > 59  and left_shoulder_angle < 63  and right_shoulder_angle > 54  and right_shoulder_angle < 58  or \
            left_shoulder_angle > 69  and left_shoulder_angle < 73  and right_shoulder_angle > 79  and right_shoulder_angle < 83  or \
            left_shoulder_angle > 78  and left_shoulder_angle < 82  and right_shoulder_angle > 94  and right_shoulder_angle < 98  or \
            left_shoulder_angle > 74  and left_shoulder_angle < 78  and right_shoulder_angle > 34  and right_shoulder_angle < 38  or \
            left_shoulder_angle > 70  and left_shoulder_angle < 74  and right_shoulder_angle > 31  and right_shoulder_angle < 35  or \
            left_shoulder_angle > 71  and left_shoulder_angle < 75  and right_shoulder_angle > 36  and right_shoulder_angle < 40  or \
            left_shoulder_angle > 53  and left_shoulder_angle < 57  and right_shoulder_angle > 8   and right_shoulder_angle < 12  or \
            left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 35  and right_shoulder_angle < 39  or \
            left_shoulder_angle > 61  and left_shoulder_angle < 65  and right_shoulder_angle > 71  and right_shoulder_angle < 75  or \
            left_shoulder_angle > 59  and left_shoulder_angle < 63  and right_shoulder_angle > 79  and right_shoulder_angle < 83  or \
            left_shoulder_angle > 44  and left_shoulder_angle < 48  and right_shoulder_angle > 81  and right_shoulder_angle < 85  or \
            left_shoulder_angle > 28  and left_shoulder_angle < 32  and right_shoulder_angle > 76  and right_shoulder_angle < 80  or \
            left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 70  and right_shoulder_angle < 74  or \
            left_shoulder_angle > 29  and left_shoulder_angle < 33  and right_shoulder_angle > 74  and right_shoulder_angle < 78  or \
            left_shoulder_angle > 28  and left_shoulder_angle < 32  and right_shoulder_angle > 77  and right_shoulder_angle < 81  or \
            left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 79  and right_shoulder_angle < 83  or \
            left_shoulder_angle > 28  and left_shoulder_angle < 32  and right_shoulder_angle > 79  and right_shoulder_angle < 83  or \
            left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 77  and right_shoulder_angle < 81  or \
            left_shoulder_angle > 55 and left_shoulder_angle < 90 and right_shoulder_angle > 45 and right_shoulder_angle < 65 :
            if left_knee_angle > 110 and left_knee_angle < 120 and right_knee_angle > 175 and right_knee_angle < 185 or \
                left_knee_angle > 0 and left_knee_angle < 20 and right_knee_angle > 170 and right_knee_angle < 185 or \
                left_knee_angle > 20 and left_knee_angle < 40 and right_knee_angle > 170 and right_knee_angle < 185 or \
                left_knee_angle > 160 and left_knee_angle < 170 and right_knee_angle > 160 and right_knee_angle < 180 or \
                left_knee_angle > 170 and left_knee_angle < 190 and right_knee_angle > 15 and right_knee_angle < 30 or \
                left_knee_angle > 170 and left_knee_angle < 190 and right_knee_angle > 40 and right_knee_angle < 50 or \
                left_knee_angle > 170 and left_knee_angle < 180 and right_knee_angle > 160 and right_knee_angle < 175 or \
                left_knee_angle > 175 and left_knee_angle < 190 and right_knee_angle > 160 and right_knee_angle < 180 or \
                left_knee_angle > 175 and left_knee_angle < 190 and right_knee_angle > 100 and right_knee_angle < 120 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 182 and right_knee_angle < 186 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 182 and right_knee_angle < 186 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 181 and right_knee_angle < 185 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 180 and right_knee_angle < 184 or \
                left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 180 and right_knee_angle < 184 or \
                left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 180 and right_knee_angle < 184 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 189 and right_knee_angle < 193 or \
                left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 185 and right_knee_angle < 189 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 188 and right_knee_angle < 192 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 190 and right_knee_angle < 194 or \
                left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 186 and right_knee_angle < 190 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 183 and right_knee_angle < 187 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 185 and right_knee_angle < 189 or \
                left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 184 and right_knee_angle < 188 or \
                left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 183 and right_knee_angle < 187 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 184 and right_knee_angle < 188 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 185 and right_knee_angle < 189 or \
                left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 184 and right_knee_angle < 188 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 183 and right_knee_angle < 187 or \
                left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 182 and right_knee_angle < 186 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 183 and right_knee_angle < 187 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 182 and right_knee_angle < 186 or \
                left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 181 and right_knee_angle < 185 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 180 and right_knee_angle < 184 or \
                left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 191 and right_knee_angle < 195 or \
                left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 192 and right_knee_angle < 196 or \
                left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 186 and right_knee_angle < 190 or \
                left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 187 and right_knee_angle < 181 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 186 and right_knee_angle < 190 or \
                left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 184 and right_knee_angle < 188 or \
                left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 182 and right_knee_angle < 186 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 185 and right_knee_angle < 189 or \
                left_knee_angle > 160 and left_knee_angle < 164 and right_knee_angle > 175 and right_knee_angle < 179 or \
                left_knee_angle > 157 and left_knee_angle < 161 and right_knee_angle > 172 and right_knee_angle < 176 or \
                left_knee_angle > 152 and left_knee_angle < 154 and right_knee_angle > 171 and right_knee_angle < 175 or \
                left_knee_angle > 159 and left_knee_angle < 163 and right_knee_angle > 164 and right_knee_angle < 168 or \
                left_knee_angle > 154 and left_knee_angle < 158 and right_knee_angle > 160 and right_knee_angle < 164 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 175 and right_knee_angle < 179 or \
                left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 189 and right_knee_angle < 193 or \
                left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 190 and right_knee_angle < 194 or \
                left_knee_angle > 190 and left_knee_angle < 194 and right_knee_angle > 195 and right_knee_angle < 199 or \
                left_knee_angle > 193 and left_knee_angle < 197 and right_knee_angle > 182 and right_knee_angle < 186 or \
                left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 176 and right_knee_angle < 180 or \
                left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 175 and right_knee_angle < 179 or \
                left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 173 and right_knee_angle < 177 or \
                left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 173 and right_knee_angle < 177 or \
                left_knee_angle > 156 and left_knee_angle < 160 and right_knee_angle > 172 and right_knee_angle < 176 or \
                left_knee_angle > 157 and left_knee_angle < 161 and right_knee_angle > 173 and right_knee_angle < 177 or \
                left_knee_angle > 160 and left_knee_angle < 164 and right_knee_angle > 163 and right_knee_angle < 167 or \
                left_knee_angle > 162 and left_knee_angle < 166 and right_knee_angle > 166 and right_knee_angle < 170 or \
                left_knee_angle > 159 and left_knee_angle < 163 and right_knee_angle > 161 and right_knee_angle < 165 or \
                left_knee_angle > 165 and left_knee_angle < 169 and right_knee_angle > 159 and right_knee_angle < 163 or \
                left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 170 and right_knee_angle < 174 or \
                left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 178 and right_knee_angle < 182 or \
                left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 190 and right_knee_angle < 194 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 191 and right_knee_angle < 195 or \
                left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 193 and right_knee_angle < 197 or \
                left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 184 and right_knee_angle < 188 or \
                left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 180 and right_knee_angle < 184 or \
                left_knee_angle > 167 and left_knee_angle < 171 and right_knee_angle > 177 and right_knee_angle < 181 or \
                left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 181 and right_knee_angle < 185 or \
                left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 178 and right_knee_angle < 182 or \
                left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 181 and right_knee_angle < 185 or \
                left_knee_angle > 165 and left_knee_angle < 169 and right_knee_angle > 180 and right_knee_angle < 184 or \
                left_knee_angle > 165 and left_knee_angle < 169 and right_knee_angle > 179 and right_knee_angle < 183 or \
                left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 174 and right_knee_angle < 178 or \
                left_knee_angle > 167 and left_knee_angle < 171 and right_knee_angle > 174 and right_knee_angle < 178 or \
                left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 171 and right_knee_angle < 175 or \
                left_knee_angle > 162 and left_knee_angle < 166 and right_knee_angle > 168 and right_knee_angle < 172 or \
                left_knee_angle > 153 and left_knee_angle < 157 and right_knee_angle > 164 and right_knee_angle < 168 or \
                left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 174 and right_knee_angle < 178 or \
                left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 172 and right_knee_angle < 176 or \
                left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 174 and right_knee_angle < 178 or \
                left_knee_angle > 186 and left_knee_angle < 190 and right_knee_angle > 186 and right_knee_angle < 190 or \
                left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 190 and right_knee_angle < 194 or \
                left_knee_angle > 183 and left_knee_angle < 187 and right_knee_angle > 209 and right_knee_angle < 213 or \
                left_knee_angle > 188 and left_knee_angle < 192 and right_knee_angle > 223 and right_knee_angle < 227 or \
                left_knee_angle > 187 and left_knee_angle < 191 and right_knee_angle > 183 and right_knee_angle < 187 or \
                left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 183 and right_knee_angle < 187 or \
                left_knee_angle > 185 and left_knee_angle < 189 and right_knee_angle > 181 and right_knee_angle < 185 or \
                left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 183 and right_knee_angle < 187 or \
                left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 178 and right_knee_angle < 182 or \
                left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 171 and right_knee_angle < 175 or \
                left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 172 and right_knee_angle < 176 or \
                left_knee_angle > 134 and left_knee_angle < 138 and right_knee_angle > 173 and right_knee_angle < 177 or \
                left_knee_angle > 64  and left_knee_angle < 68  and right_knee_angle > 169 and right_knee_angle < 173 or \
                left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 178 and right_knee_angle < 182 or \
                left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 179 and right_knee_angle < 183 or \
                left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 180 and right_knee_angle < 184 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 181 and right_knee_angle < 185 or \
                left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 190 and right_knee_angle < 194 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 233 and right_knee_angle < 237 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 194 and right_knee_angle < 198 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 184 and right_knee_angle < 188 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 179 and right_knee_angle < 183 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 172 and right_knee_angle < 176 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 174 and right_knee_angle < 178 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 177 and right_knee_angle < 181 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 169 and right_knee_angle < 173 or \
                left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 170 and right_knee_angle < 174 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 177 and right_knee_angle < 181 or \
                left_knee_angle > 335 and left_knee_angle < 350 and right_knee_angle > 170 and right_knee_angle < 180 :
                label = "Ketrib_Jaroe"   
                last_label = label 

        if left_elbow_angle > 125 and left_elbow_angle < 145 and right_elbow_angle > 230 and right_elbow_angle < 260 or \
        left_elbow_angle > 120 and left_elbow_angle < 170 and right_elbow_angle > 205 and right_elbow_angle < 240 or \
        left_elbow_angle > 155 and left_elbow_angle < 170 and right_elbow_angle > 205 and right_elbow_angle < 230 or \
        left_elbow_angle > 125 and left_elbow_angle < 150 and right_elbow_angle > 210 and right_elbow_angle < 240 or \
        left_elbow_angle > 155 and left_elbow_angle < 170 and right_elbow_angle > 190 and right_elbow_angle < 230 or \
        left_elbow_angle > 145 and left_elbow_angle < 160 and right_elbow_angle > 205 and right_elbow_angle < 230 or \
        left_elbow_angle > 145 and left_elbow_angle < 180 and right_elbow_angle > 180 and right_elbow_angle < 230 or \
        left_elbow_angle > 185 and left_elbow_angle < 210 and right_elbow_angle > 200 and right_elbow_angle < 245 or \
        left_elbow_angle > 200 and left_elbow_angle < 220 and right_elbow_angle > 270 and right_elbow_angle < 290 or \
        left_elbow_angle > 200 and left_elbow_angle < 250 and right_elbow_angle > 200 and right_elbow_angle < 285 or \
        left_elbow_angle > 200 and left_elbow_angle < 220 and right_elbow_angle > 170 and right_elbow_angle < 200 or \
        left_elbow_angle > 200 and left_elbow_angle < 220 and right_elbow_angle > 220 and right_elbow_angle < 250 or \
        left_elbow_angle > 200 and left_elbow_angle < 220 and right_elbow_angle > 200 and right_elbow_angle < 220 or \
        left_elbow_angle > 210 and left_elbow_angle < 240 and right_elbow_angle > 220 and right_elbow_angle < 250 or \
        left_elbow_angle > 210 and left_elbow_angle < 240 and right_elbow_angle > 195 and right_elbow_angle < 230 or \
        left_elbow_angle > 230 and left_elbow_angle < 260 and right_elbow_angle > 190 and right_elbow_angle < 220 or \
        left_elbow_angle > 200 and left_elbow_angle < 230 and right_elbow_angle > 275 and right_elbow_angle < 300 or \
        left_elbow_angle > 200 and left_elbow_angle < 230 and right_elbow_angle > 220 and right_elbow_angle < 245 or \
        left_elbow_angle > 50 and left_elbow_angle < 95 and right_elbow_angle > 245 and right_elbow_angle < 270 or \
        left_elbow_angle > 70 and left_elbow_angle < 95 and right_elbow_angle > 240 and right_elbow_angle < 270 or \
        left_elbow_angle > 10 and left_elbow_angle < 30 and right_elbow_angle > 220 and right_elbow_angle < 270 or \
        left_elbow_angle > 265 and left_elbow_angle < 285 and right_elbow_angle > 195 and right_elbow_angle < 210 or \
        left_elbow_angle > 170 and left_elbow_angle < 205 and right_elbow_angle > 180 and right_elbow_angle < 210 or \
        left_elbow_angle > 280 and left_elbow_angle < 295 and right_elbow_angle > 190 and right_elbow_angle < 210 or \
        left_elbow_angle > 320 and left_elbow_angle < 330 and right_elbow_angle > 210 and right_elbow_angle < 220 or \
        left_elbow_angle > 110 and left_elbow_angle < 125 and right_elbow_angle > 220 and right_elbow_angle < 230 or \
        left_elbow_angle > 65 and left_elbow_angle < 75 and right_elbow_angle > 220 and right_elbow_angle < 240 or \
        left_elbow_angle > 340 and left_elbow_angle < 350 and right_elbow_angle > 215 and right_elbow_angle < 225 or \
        left_elbow_angle > 235 and left_elbow_angle < 245 and right_elbow_angle > 210 and right_elbow_angle < 225 or \
        left_elbow_angle > 180 and left_elbow_angle < 195 and right_elbow_angle > 195 and right_elbow_angle < 200 or \
        left_elbow_angle > 245 and left_elbow_angle < 255 and right_elbow_angle > 195 and right_elbow_angle < 200 or \
        left_elbow_angle > 335 and left_elbow_angle < 340 and right_elbow_angle > 200 and right_elbow_angle < 210 or \
        left_elbow_angle > 50 and left_elbow_angle < 60 and right_elbow_angle > 230 and right_elbow_angle < 235 or \
        left_elbow_angle > 100 and left_elbow_angle < 120 and right_elbow_angle > 220 and right_elbow_angle < 230 or \
        left_elbow_angle > 80 and left_elbow_angle < 90 and right_elbow_angle > 235 and right_elbow_angle < 245 or \
        left_elbow_angle > 15 and left_elbow_angle < 25 and right_elbow_angle > 210 and right_elbow_angle < 225 or \
        left_elbow_angle > 320 and left_elbow_angle < 330 and right_elbow_angle > 200 and right_elbow_angle < 210 or \
        left_elbow_angle > 215 and left_elbow_angle < 220 and right_elbow_angle > 180 and right_elbow_angle < 190 or \
        left_elbow_angle > 180 and left_elbow_angle < 195 and right_elbow_angle > 180 and right_elbow_angle < 190 or \
        left_elbow_angle > 180 and left_elbow_angle < 195 and right_elbow_angle > 180 and right_elbow_angle < 200 or \
        left_elbow_angle > 270 and left_elbow_angle < 280 and right_elbow_angle > 200 and right_elbow_angle < 205 or \
        left_elbow_angle > 95 and left_elbow_angle < 105 and right_elbow_angle > 265 and right_elbow_angle < 275 or \
        left_elbow_angle > 115 and left_elbow_angle < 120 and right_elbow_angle > 270 and right_elbow_angle < 280 or \
        left_elbow_angle > 155 and left_elbow_angle < 160 and right_elbow_angle >225 and right_elbow_angle < 235 or \
        left_elbow_angle > 155 and left_elbow_angle < 165 and right_elbow_angle >215 and right_elbow_angle < 220 or \
        left_elbow_angle > 155 and left_elbow_angle < 160 and right_elbow_angle >195 and right_elbow_angle < 200 or \
        left_elbow_angle > 145 and left_elbow_angle < 150 and right_elbow_angle >205 and right_elbow_angle < 210 or \
        left_elbow_angle > 115 and left_elbow_angle < 120 and right_elbow_angle >230 and right_elbow_angle < 235 or \
        left_elbow_angle > 140 and left_elbow_angle < 145 and right_elbow_angle >210 and right_elbow_angle < 220 or \
        left_elbow_angle > 145 and left_elbow_angle < 150 and right_elbow_angle >195 and right_elbow_angle < 205 or \
        left_elbow_angle > 140 and left_elbow_angle < 150 and right_elbow_angle >205 and right_elbow_angle < 210 or \
        left_elbow_angle > 120 and left_elbow_angle < 125 and right_elbow_angle >220 and right_elbow_angle < 225 or \
        left_elbow_angle > 150 and left_elbow_angle < 155 and right_elbow_angle >205 and right_elbow_angle < 215 or \
        left_elbow_angle > 120 and left_elbow_angle < 130 and right_elbow_angle >225 and right_elbow_angle < 230 or \
        left_elbow_angle > 110 and left_elbow_angle < 120 and right_elbow_angle >220 and right_elbow_angle < 230 or \
        left_elbow_angle > 135 and left_elbow_angle < 145 and right_elbow_angle >210 and right_elbow_angle < 215 or \
        left_elbow_angle > 130 and left_elbow_angle < 140 and right_elbow_angle >205 and right_elbow_angle < 215 or \
        left_elbow_angle > 118 and left_elbow_angle < 125 and right_elbow_angle >236 and right_elbow_angle < 248 or \
        left_elbow_angle > 135 and left_elbow_angle < 230 and right_elbow_angle >220 and right_elbow_angle < 225 or \
        left_elbow_angle > 145 and left_elbow_angle < 255 and right_elbow_angle >200 and right_elbow_angle < 210 or \
        left_elbow_angle > 95 and left_elbow_angle < 105 and right_elbow_angle > 245 and right_elbow_angle < 250 or \
        left_elbow_angle > 115 and left_elbow_angle < 120 and right_elbow_angle > 235 and right_elbow_angle < 240 or \
        left_elbow_angle > 324 and left_elbow_angle < 330 and right_elbow_angle > 305 and right_elbow_angle < 310 or \
        left_elbow_angle > 5 and left_elbow_angle < 10 and right_elbow_angle > 275 and right_elbow_angle < 280 or \
        left_elbow_angle > 70 and left_elbow_angle < 75 and right_elbow_angle > 235 and right_elbow_angle < 240 or \
        left_elbow_angle > 74 and left_elbow_angle < 89 and right_elbow_angle > 235 and right_elbow_angle < 240 or \
        left_elbow_angle > 95 and left_elbow_angle < 100 and right_elbow_angle > 234 and right_elbow_angle < 240 or \
        left_elbow_angle > 130 and left_elbow_angle < 135 and right_elbow_angle > 220 and right_elbow_angle < 225 or \
        left_elbow_angle > 155 and left_elbow_angle < 160 and right_elbow_angle > 200 and right_elbow_angle < 205 or \
        left_elbow_angle > 165 and left_elbow_angle < 170 and right_elbow_angle > 175 and right_elbow_angle < 180 or \
        left_elbow_angle > 170 and left_elbow_angle < 175 and right_elbow_angle > 179 and right_elbow_angle < 185 or \
        left_elbow_angle > 143 and left_elbow_angle < 152 and right_elbow_angle > 205 and right_elbow_angle < 211 or \
        left_elbow_angle > 165 and left_elbow_angle < 173 and right_elbow_angle > 190 and right_elbow_angle < 198 or \
        left_elbow_angle > 148 and left_elbow_angle < 163 and right_elbow_angle > 215 and right_elbow_angle < 238 or \
        left_elbow_angle > 165 and left_elbow_angle < 175 and right_elbow_angle > 200 and right_elbow_angle < 205 or \
        left_elbow_angle > 161 and left_elbow_angle < 151 and right_elbow_angle > 225 and right_elbow_angle < 231 or \
        left_elbow_angle > 139 and left_elbow_angle < 143 and right_elbow_angle > 215 and right_elbow_angle < 229 or \
        left_elbow_angle > 170 and left_elbow_angle < 181 and right_elbow_angle > 187 and right_elbow_angle < 196 or \
        left_elbow_angle > 175 and left_elbow_angle < 180 and right_elbow_angle > 188 and right_elbow_angle < 196 or \
        left_elbow_angle > 180 and left_elbow_angle < 183 and right_elbow_angle > 239 and right_elbow_angle < 245 or \
        left_elbow_angle > 180 and left_elbow_angle < 183 and right_elbow_angle > 205 and right_elbow_angle < 210 or \
        left_elbow_angle > 170 and left_elbow_angle < 177 and right_elbow_angle > 169 and right_elbow_angle < 171 or \
        left_elbow_angle > 164 and left_elbow_angle < 167 and right_elbow_angle > 175 and right_elbow_angle < 182 or \
        left_elbow_angle > 155 and left_elbow_angle < 170 and right_elbow_angle > 235 and right_elbow_angle < 242 or \
        left_elbow_angle > 155 and left_elbow_angle < 170 and right_elbow_angle > 240 and right_elbow_angle < 253 or \
        left_elbow_angle > 147 and left_elbow_angle < 153 and right_elbow_angle > 195 and right_elbow_angle < 200 or \
        left_elbow_angle > 158 and left_elbow_angle < 165 and right_elbow_angle > 192 and right_elbow_angle < 199 or \
        left_elbow_angle > 153 and left_elbow_angle < 158 and right_elbow_angle > 192 and right_elbow_angle < 199 or \
        left_elbow_angle > 157 and left_elbow_angle < 163 and right_elbow_angle > 185 and right_elbow_angle < 199 or \
        left_elbow_angle > 158 and left_elbow_angle < 164 and right_elbow_angle > 183 and right_elbow_angle < 187 or \
        left_elbow_angle > 157 and left_elbow_angle < 163 and right_elbow_angle > 185 and right_elbow_angle < 193 or \
        left_elbow_angle > 124 and left_elbow_angle < 130 and right_elbow_angle > 200 and right_elbow_angle < 207 or \
        left_elbow_angle > 113 and left_elbow_angle < 117 and right_elbow_angle > 204 and right_elbow_angle < 208 or \
        left_elbow_angle > 130 and left_elbow_angle < 135 and right_elbow_angle > 189 and right_elbow_angle < 195 or \
        left_elbow_angle > 197 and left_elbow_angle < 205 and right_elbow_angle > 194 and right_elbow_angle < 197 or \
        left_elbow_angle > 110 and left_elbow_angle < 117 and right_elbow_angle > 210 and right_elbow_angle < 219 or \
        left_elbow_angle > 110 and left_elbow_angle < 117 and right_elbow_angle > 203 and right_elbow_angle < 207 or \
        left_elbow_angle > 200 and left_elbow_angle < 205 and right_elbow_angle > 199 and right_elbow_angle < 204 or \
        left_elbow_angle > 180 and left_elbow_angle < 193 and right_elbow_angle > 181 and right_elbow_angle < 193 or \
        left_elbow_angle > 133 and left_elbow_angle < 140 and right_elbow_angle > 215 and right_elbow_angle < 235 or \
        left_elbow_angle > 150 and left_elbow_angle < 160 and right_elbow_angle > 198 and right_elbow_angle < 203 or \
        left_elbow_angle > 121 and left_elbow_angle < 125 and right_elbow_angle > 222 and right_elbow_angle < 226 or \
        left_elbow_angle > 151 and left_elbow_angle < 161 and right_elbow_angle > 191 and right_elbow_angle < 199 or \
        left_elbow_angle > 145 and left_elbow_angle < 149 and right_elbow_angle > 215 and right_elbow_angle < 220 or \
        left_elbow_angle > 152 and left_elbow_angle < 158 and right_elbow_angle > 199 and right_elbow_angle < 203 or \
        left_elbow_angle > 159 and left_elbow_angle < 162 and right_elbow_angle > 195 and right_elbow_angle < 208 or \
        left_elbow_angle > 150 and left_elbow_angle < 162 and right_elbow_angle > 195 and right_elbow_angle < 208 or \
        left_elbow_angle > 142 and left_elbow_angle < 146 and right_elbow_angle > 203 and right_elbow_angle < 207 or \
        left_elbow_angle > 153 and left_elbow_angle < 163 and right_elbow_angle > 190 and right_elbow_angle < 218 or \
        left_elbow_angle > 310 and left_elbow_angle < 330 and right_elbow_angle > 220 and right_elbow_angle < 240:
            if left_shoulder_angle > 10 and left_shoulder_angle < 30 and right_shoulder_angle > 10 and right_shoulder_angle < 40 or \
                left_shoulder_angle > 0 and left_shoulder_angle < 20 and right_shoulder_angle > 45 and right_shoulder_angle < 65 or \
                left_shoulder_angle > 0 and left_shoulder_angle < 20 and right_shoulder_angle > 20 and right_shoulder_angle < 25 or \
                left_shoulder_angle > 0 and left_shoulder_angle < 20 and right_shoulder_angle > 70 and right_shoulder_angle < 90 or \
                left_shoulder_angle > 0 and left_shoulder_angle < 20 and right_shoulder_angle > 15 and right_shoulder_angle < 35 or \
                left_shoulder_angle > 50 and left_shoulder_angle < 65 and right_shoulder_angle > 50 and right_shoulder_angle < 60 or \
                left_shoulder_angle > 30 and left_shoulder_angle < 40 and right_shoulder_angle > 40 and right_shoulder_angle < 60 or \
                left_shoulder_angle > 40 and left_shoulder_angle < 60 and right_shoulder_angle > 45 and right_shoulder_angle < 65 or \
                left_shoulder_angle > 25 and left_shoulder_angle < 50 and right_shoulder_angle > 30 and right_shoulder_angle < 50 or \
                left_shoulder_angle > 25 and left_shoulder_angle < 50 and right_shoulder_angle > 40 and right_shoulder_angle < 50 or \
                left_shoulder_angle > 25 and left_shoulder_angle < 50 and right_shoulder_angle > 20 and right_shoulder_angle < 40 or \
                left_shoulder_angle > 0 and left_shoulder_angle < 10 and right_shoulder_angle > 10 and right_shoulder_angle < 20 or \
                left_shoulder_angle > 350 and left_shoulder_angle < 360 and right_shoulder_angle > 0 and right_shoulder_angle < 10 or \
                left_shoulder_angle > 25 and left_shoulder_angle < 30 and right_shoulder_angle > 20 and right_shoulder_angle < 25 or \
                left_shoulder_angle > 25 and left_shoulder_angle < 40 and right_shoulder_angle > 30 and right_shoulder_angle < 50 or \
                left_shoulder_angle > 20 and left_shoulder_angle < 50 and right_shoulder_angle > 20 and right_shoulder_angle < 50 or \
                left_shoulder_angle > 30 and left_shoulder_angle < 45 and right_shoulder_angle > 30 and right_shoulder_angle < 45 or \
                left_shoulder_angle > 40 and left_shoulder_angle < 60 and right_shoulder_angle > 60 and right_shoulder_angle < 85 or \
                left_shoulder_angle > 160 and left_shoulder_angle < 190 and right_shoulder_angle > 30 and right_shoulder_angle < 50 or \
                left_shoulder_angle > 340 and left_shoulder_angle < 360 and right_shoulder_angle > 55 and right_shoulder_angle < 75 or \
                left_shoulder_angle > 350 and left_shoulder_angle < 360 and right_shoulder_angle > 0 and right_shoulder_angle < 20 or \
                left_shoulder_angle > 40 and left_shoulder_angle < 50 and right_shoulder_angle > 35 and right_shoulder_angle < 45 or \
                left_shoulder_angle > 30 and left_shoulder_angle < 35 and right_shoulder_angle > 25 and right_shoulder_angle < 35 or \
                left_shoulder_angle > 38 and left_shoulder_angle < 45 and right_shoulder_angle > 38 and right_shoulder_angle < 50 or \
                left_shoulder_angle > 35 and left_shoulder_angle < 45 and right_shoulder_angle > 30 and right_shoulder_angle < 40 or \
                left_shoulder_angle > 30 and left_shoulder_angle < 45 and right_shoulder_angle > 30 and right_shoulder_angle < 40 or \
                left_shoulder_angle > 330 and left_shoulder_angle < 360 and right_shoulder_angle > 26 and right_shoulder_angle < 44 or \
                left_shoulder_angle > 33 and left_shoulder_angle < 41 and right_shoulder_angle > 40 and right_shoulder_angle < 50 or \
                left_shoulder_angle > 20 and left_shoulder_angle < 25 and right_shoulder_angle > 30 and right_shoulder_angle < 35 or \
                left_shoulder_angle > 49 and left_shoulder_angle < 58 and right_shoulder_angle > 58 and right_shoulder_angle < 65 or \
                left_shoulder_angle > 41 and left_shoulder_angle < 43 and right_shoulder_angle > 45 and right_shoulder_angle < 47 or \
                left_shoulder_angle > 34 and left_shoulder_angle < 36 and right_shoulder_angle > 44 and right_shoulder_angle < 46 or \
                left_shoulder_angle > 28 and left_shoulder_angle < 30 and right_shoulder_angle > 28 and right_shoulder_angle < 31 or \
                left_shoulder_angle > 17 and left_shoulder_angle < 23 and right_shoulder_angle > 5 and right_shoulder_angle < 15 or \
                left_shoulder_angle > 45 and left_shoulder_angle < 61 and right_shoulder_angle > 55 and right_shoulder_angle < 65 or \
                left_shoulder_angle > 34 and left_shoulder_angle < 42 and right_shoulder_angle > 35 and right_shoulder_angle < 51 or \
                left_shoulder_angle >  5 and left_shoulder_angle < 20 and right_shoulder_angle > 5 and right_shoulder_angle < 10 or \
                left_shoulder_angle >  5 and left_shoulder_angle < 20 and right_shoulder_angle > 355 and right_shoulder_angle < 360 or \
                left_shoulder_angle >  5 and left_shoulder_angle < 20 and right_shoulder_angle > 0 and right_shoulder_angle < 5 or \
                left_shoulder_angle >  27 and left_shoulder_angle < 34 and right_shoulder_angle > 19 and right_shoulder_angle < 25 or \
                left_shoulder_angle >  43 and left_shoulder_angle < 45 and right_shoulder_angle > 30 and right_shoulder_angle < 35 or \
                left_shoulder_angle > 30 and left_shoulder_angle < 35 and right_shoulder_angle > 25 and right_shoulder_angle < 30 or \
                left_shoulder_angle > 45 and left_shoulder_angle < 70 and right_shoulder_angle > 39 and right_shoulder_angle < 60 or \
                left_shoulder_angle > 45 and left_shoulder_angle < 70 and right_shoulder_angle > 39 and right_shoulder_angle < 66 or \
                left_shoulder_angle > 330 and left_shoulder_angle < 360 and right_shoulder_angle > 25 and right_shoulder_angle < 60 :
                if left_knee_angle > 150 and left_knee_angle < 170 and right_knee_angle > 150 and right_knee_angle < 185 or \
                    left_knee_angle > 155 and left_knee_angle < 180 and right_knee_angle > 125 and right_knee_angle < 160 or \
                    left_knee_angle > 170 and left_knee_angle < 190 and right_knee_angle > 165 and right_knee_angle < 190 or \
                    left_knee_angle > 145 and left_knee_angle < 190 and right_knee_angle > 165 and right_knee_angle < 190 or \
                    left_knee_angle > 165 and left_knee_angle < 170 and right_knee_angle > 160 and right_knee_angle < 165 or \
                    left_knee_angle > 150 and left_knee_angle < 160 and right_knee_angle > 155 and right_knee_angle < 160 or \
                    left_knee_angle > 110 and left_knee_angle < 125 and right_knee_angle > 165 and right_knee_angle < 175 or \
                    left_knee_angle > 105 and left_knee_angle < 125 and right_knee_angle > 160 and right_knee_angle < 175 or \
                    left_knee_angle > 180 and left_knee_angle < 190 and right_knee_angle > 145 and right_knee_angle < 160 or \
                    left_knee_angle > 172 and left_knee_angle < 180 and right_knee_angle > 140 and right_knee_angle < 155 or \
                    left_knee_angle > 150 and left_knee_angle < 170 and right_knee_angle > 140 and right_knee_angle < 150 or \
                    left_knee_angle > 175 and left_knee_angle < 180 and right_knee_angle > 155 and right_knee_angle < 160 or \
                    left_knee_angle > 170 and left_knee_angle < 185 and right_knee_angle > 140 and right_knee_angle < 155 or \
                    left_knee_angle > 160 and left_knee_angle < 170 and right_knee_angle > 150 and right_knee_angle < 155 or \
                    left_knee_angle > 135 and left_knee_angle < 140 and right_knee_angle > 145 and right_knee_angle <  155 or \
                    left_knee_angle > 160 and left_knee_angle < 175 and right_knee_angle > 145 and right_knee_angle <  155 or \
                    left_knee_angle > 175 and left_knee_angle < 180 and right_knee_angle > 135 and right_knee_angle <  160 or \
                    left_knee_angle > 140 and left_knee_angle < 155 and right_knee_angle > 100 and right_knee_angle < 130 or \
                    left_knee_angle > 130 and left_knee_angle < 135 and right_knee_angle > 165 and right_knee_angle < 170 or \
                    left_knee_angle > 115 and left_knee_angle < 120 and right_knee_angle > 170 and right_knee_angle < 175 or \
                    left_knee_angle > 154 and left_knee_angle < 167 and right_knee_angle > 140 and right_knee_angle < 160 or \
                    left_knee_angle > 148 and left_knee_angle < 150 and right_knee_angle > 130 and right_knee_angle < 135 or \
                    left_knee_angle > 130 and left_knee_angle < 135 and right_knee_angle > 115 and right_knee_angle < 120 or \
                    left_knee_angle > 100 and left_knee_angle < 105 and right_knee_angle > 124 and right_knee_angle < 129 or \
                    left_knee_angle > 165 and left_knee_angle < 170 and right_knee_angle > 125 and right_knee_angle < 129 or \
                    left_knee_angle > 155 and left_knee_angle < 175 and right_knee_angle > 110 and right_knee_angle < 115 or \
                    left_knee_angle > 90 and left_knee_angle < 107 and right_knee_angle > 161 and right_knee_angle < 165 or \
                    left_knee_angle > 145 and left_knee_angle < 180 and right_knee_angle > 133 and right_knee_angle < 142 or \
                    left_knee_angle > 145 and left_knee_angle < 180 and right_knee_angle > 105 and right_knee_angle < 115 or \
                    left_knee_angle > 145 and left_knee_angle < 180 and right_knee_angle > 135 and right_knee_angle < 146 or \
                    left_knee_angle > 131 and left_knee_angle < 135 and right_knee_angle > 135 and right_knee_angle < 139 or \
                    left_knee_angle > 151 and left_knee_angle < 155 and right_knee_angle > 151 and right_knee_angle < 155 or \
                    left_knee_angle > 150 and left_knee_angle < 154 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 154 and left_knee_angle < 158 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 154 and left_knee_angle < 157 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 153 and left_knee_angle < 157 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 180 and left_knee_angle < 185 and right_knee_angle > 166 and right_knee_angle < 170 or \
                    left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 149 and right_knee_angle <  155 or \
                    left_knee_angle > 163 and left_knee_angle < 166 and right_knee_angle > 145 and right_knee_angle <  149 or \
                    left_knee_angle > 172 and left_knee_angle < 182 and right_knee_angle > 160 and right_knee_angle <  165 or \
                    left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 149 and right_knee_angle <  152 or \
                    left_knee_angle > 156 and left_knee_angle < 163 and right_knee_angle > 141 and right_knee_angle <  150 or \
                    left_knee_angle > 151 and left_knee_angle < 155 and right_knee_angle > 151 and right_knee_angle < 155 or \
                    left_knee_angle > 150 and left_knee_angle < 154 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 154 and left_knee_angle < 158 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 154 and left_knee_angle < 157 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 153 and left_knee_angle < 157 and right_knee_angle > 143 and right_knee_angle < 147 or \
                    left_knee_angle > 180 and left_knee_angle < 185 and right_knee_angle > 166 and right_knee_angle < 170 or \
                    left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 149 and right_knee_angle <  155 or \
                    left_knee_angle > 163 and left_knee_angle < 166 and right_knee_angle > 145 and right_knee_angle <  149 or \
                    left_knee_angle > 172 and left_knee_angle < 182 and right_knee_angle > 160 and right_knee_angle <  165 or \
                    left_knee_angle > 158 and left_knee_angle < 163 and right_knee_angle > 151 and right_knee_angle < 155 or \
                    left_knee_angle > 145 and left_knee_angle < 148 and right_knee_angle > 150 and right_knee_angle <  155 or \
                    left_knee_angle > 140 and left_knee_angle < 145 and right_knee_angle > 152 and right_knee_angle <  156 or \
                    left_knee_angle > 143 and left_knee_angle < 147 and right_knee_angle > 151 and right_knee_angle <  155 or \
                    left_knee_angle > 165 and left_knee_angle < 169 and right_knee_angle > 144 and right_knee_angle <  148 or \
                    left_knee_angle > 180 and left_knee_angle < 192 and right_knee_angle > 155 and right_knee_angle <  169 or \
                    left_knee_angle > 184 and left_knee_angle < 186 and right_knee_angle > 156 and right_knee_angle <  161 or \
                    left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 152 and right_knee_angle <  156 or \
                    left_knee_angle > 146 and left_knee_angle < 148 and right_knee_angle > 154 and right_knee_angle <  158 or \
                    left_knee_angle > 134 and left_knee_angle < 137 and right_knee_angle > 160 and right_knee_angle <  164 or \
                    left_knee_angle > 119 and left_knee_angle < 123 and right_knee_angle > 159 and right_knee_angle <  163 or \
                    left_knee_angle > 145 and left_knee_angle < 148 and right_knee_angle > 156 and right_knee_angle <  159 or \
                    left_knee_angle > 167 and left_knee_angle < 170 and right_knee_angle > 151 and right_knee_angle <  155 or \
                    left_knee_angle > 136 and left_knee_angle < 140 and right_knee_angle > 159 and right_knee_angle <  163 or \
                    left_knee_angle > 132 and left_knee_angle < 136 and right_knee_angle > 151 and right_knee_angle <  155 or \
                    left_knee_angle > 139 and left_knee_angle < 143 and right_knee_angle > 158 and right_knee_angle <  163 or \
                    left_knee_angle > 167 and left_knee_angle < 173 and right_knee_angle > 140 and right_knee_angle <  159 or \
                    left_knee_angle > 154 and left_knee_angle < 158 and right_knee_angle > 140 and right_knee_angle <  159 or \
                    left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 149 and right_knee_angle <  152 or \
                    left_knee_angle > 156 and left_knee_angle < 163 and right_knee_angle > 141 and right_knee_angle <  150 or \
                    left_knee_angle > 140 and left_knee_angle < 155 and right_knee_angle > 100 and right_knee_angle < 130 :
                    label = "Hayak_Baho"
                    last_label = label

    if left_elbow_angle > 160 and left_elbow_angle < 180 and right_elbow_angle > 290 and right_elbow_angle < 310 or \
        left_elbow_angle > 160 and left_elbow_angle < 180 and right_elbow_angle > 275 and right_elbow_angle < 290 or \
        left_elbow_angle > 160 and left_elbow_angle < 180 and right_elbow_angle > 180 and right_elbow_angle < 200 or \
        left_elbow_angle > 180 and left_elbow_angle < 200 and right_elbow_angle > 190 and right_elbow_angle < 210 or \
        left_elbow_angle > 200 and left_elbow_angle < 230 and right_elbow_angle > 320 and right_elbow_angle < 350 or \
        left_elbow_angle > 190 and left_elbow_angle < 230 and right_elbow_angle > 120 and right_elbow_angle < 145 or \
        left_elbow_angle > 0 and left_elbow_angle < 10 and right_elbow_angle > 340 and right_elbow_angle < 360 or \
        left_elbow_angle > 270 and left_elbow_angle < 290 and right_elbow_angle > 315 and right_elbow_angle < 330 or \
        left_elbow_angle > 280 and left_elbow_angle < 310 and right_elbow_angle > 30 and right_elbow_angle < 60 or \
        left_elbow_angle > 260 and left_elbow_angle < 270 and right_elbow_angle > 325 and right_elbow_angle < 340 or \
        left_elbow_angle > 240 and left_elbow_angle < 260 and right_elbow_angle > 320 and right_elbow_angle < 340 or \
        left_elbow_angle > 230 and left_elbow_angle < 250 and right_elbow_angle > 330 and right_elbow_angle < 350 or \
        left_elbow_angle > 230 and left_elbow_angle < 250 and right_elbow_angle > 320 and right_elbow_angle < 350 or \
        left_elbow_angle > 345 and left_elbow_angle < 360 and right_elbow_angle > 345 and right_elbow_angle < 360 or \
        left_elbow_angle > 150 and left_elbow_angle < 170 and right_elbow_angle > 180 and right_elbow_angle < 210 or \
        left_elbow_angle > 184 and left_elbow_angle < 188 and right_elbow_angle > 320 and right_elbow_angle < 324 or \
        left_elbow_angle > 184 and left_elbow_angle < 188 and right_elbow_angle > 300 and right_elbow_angle < 310 or \
        left_elbow_angle > 173 and left_elbow_angle < 177 and right_elbow_angle > 300 and right_elbow_angle < 310 or \
        left_elbow_angle > 318 and left_elbow_angle < 322 and right_elbow_angle > 300 and right_elbow_angle < 310 or \
        left_elbow_angle > 259 and left_elbow_angle < 273 and right_elbow_angle > 330 and right_elbow_angle < 340 or \
        left_elbow_angle > 267 and left_elbow_angle < 280 and right_elbow_angle > 330 and right_elbow_angle < 340 or \
        left_elbow_angle > 217 and left_elbow_angle < 221 and right_elbow_angle > 308 and right_elbow_angle < 312 or \
        left_elbow_angle > 179 and left_elbow_angle < 185 and right_elbow_angle > 261 and right_elbow_angle < 265 or \
        left_elbow_angle > 179 and left_elbow_angle < 185 and right_elbow_angle > 226 and right_elbow_angle < 230 or \
        left_elbow_angle > 178 and left_elbow_angle < 183 and right_elbow_angle > 197 and right_elbow_angle < 200 or \
        left_elbow_angle > 175 and left_elbow_angle < 183 and right_elbow_angle > 194 and right_elbow_angle < 200 or \
        left_elbow_angle > 180 and left_elbow_angle < 184 and right_elbow_angle > 198 and right_elbow_angle < 205 or \
        left_elbow_angle > 177 and left_elbow_angle < 183 and right_elbow_angle > 191 and right_elbow_angle < 205 or \
        left_elbow_angle > 177 and left_elbow_angle < 183 and right_elbow_angle > 204 and right_elbow_angle < 208 or \
        left_elbow_angle > 177 and left_elbow_angle < 183 and right_elbow_angle > 184 and right_elbow_angle < 190 or \
        left_elbow_angle > 179 and left_elbow_angle < 182 and right_elbow_angle > 253 and right_elbow_angle < 257 or \
        left_elbow_angle > 179 and left_elbow_angle < 182 and right_elbow_angle > 314 and right_elbow_angle < 318 or \
        left_elbow_angle > 173 and left_elbow_angle < 178 and right_elbow_angle > 313 and right_elbow_angle < 316 or \
        left_elbow_angle > 140 and left_elbow_angle < 144 and right_elbow_angle > 310 and right_elbow_angle < 314 or \
        left_elbow_angle > 270 and left_elbow_angle < 274 and right_elbow_angle > 330 and right_elbow_angle < 334 or \
        left_elbow_angle > 273 and left_elbow_angle < 277 and right_elbow_angle > 335 and right_elbow_angle < 339 or \
        left_elbow_angle > 279 and left_elbow_angle < 283 and right_elbow_angle > 342 and right_elbow_angle < 348 or \
        left_elbow_angle > 125 and left_elbow_angle < 129 and right_elbow_angle > 252 and right_elbow_angle < 258 or \
        left_elbow_angle > 174 and left_elbow_angle < 179 and right_elbow_angle > 180 and right_elbow_angle < 199 or \
        left_elbow_angle > 173 and left_elbow_angle < 179 and right_elbow_angle > 184 and right_elbow_angle < 189 or \
        left_elbow_angle > 5 and left_elbow_angle < 10 and right_elbow_angle > 345 and right_elbow_angle < 349 or \
        left_elbow_angle > 196 and left_elbow_angle < 200 and right_elbow_angle > 262 and right_elbow_angle < 275 or \
        left_elbow_angle > 232 and left_elbow_angle < 236 and right_elbow_angle > 297 and right_elbow_angle < 301 or \
        left_elbow_angle > 250 and left_elbow_angle < 257 and right_elbow_angle > 316 and right_elbow_angle < 320 or \
        left_elbow_angle > 249 and left_elbow_angle < 257 and right_elbow_angle > 355 and right_elbow_angle < 359 or \
        left_elbow_angle > 302 and left_elbow_angle < 306 and right_elbow_angle > 340 and right_elbow_angle < 344 or \
        left_elbow_angle > 186 and left_elbow_angle < 190 and right_elbow_angle > 207 and right_elbow_angle < 211 or \
        left_elbow_angle > 162 and left_elbow_angle < 166 and right_elbow_angle > 197 and right_elbow_angle < 201 or \
        left_elbow_angle > 168 and left_elbow_angle < 173 and right_elbow_angle > 174 and right_elbow_angle < 186 or \
        left_elbow_angle > 163 and left_elbow_angle < 176 and right_elbow_angle > 177 and right_elbow_angle < 190 or \
        left_elbow_angle > 163 and left_elbow_angle < 176 and right_elbow_angle > 177 and right_elbow_angle < 192 or \
        left_elbow_angle > 140 and left_elbow_angle < 144 and right_elbow_angle > 197 and right_elbow_angle < 201 or \
        left_elbow_angle > 140 and left_elbow_angle < 144 and right_elbow_angle > 287 and right_elbow_angle < 291 or \
        left_elbow_angle > 71 and left_elbow_angle < 75 and right_elbow_angle > 296 and right_elbow_angle < 300 or \
        left_elbow_angle > 40 and left_elbow_angle < 46 and right_elbow_angle > 302 and right_elbow_angle < 306 or \
        left_elbow_angle > 318 and left_elbow_angle < 322 and right_elbow_angle > 319 and right_elbow_angle < 322 or \
        left_elbow_angle > 274 and left_elbow_angle < 278 and right_elbow_angle > 331 and right_elbow_angle < 335 or \
        left_elbow_angle > 239 and left_elbow_angle < 242 and right_elbow_angle > 0 and right_elbow_angle < 5 or \
        left_elbow_angle > 249 and left_elbow_angle < 258 and right_elbow_angle > 339 and right_elbow_angle < 349 or \
        left_elbow_angle > 249 and left_elbow_angle < 258 and right_elbow_angle > 310 and right_elbow_angle < 314 or \
        left_elbow_angle > 269 and left_elbow_angle < 284 and right_elbow_angle > 349 and right_elbow_angle < 354 or \
        left_elbow_angle > 271 and left_elbow_angle < 275 and right_elbow_angle > 200 and right_elbow_angle < 204 or \
        left_elbow_angle > 169 and left_elbow_angle < 177 and right_elbow_angle > 182 and right_elbow_angle < 188 or \
        left_elbow_angle > 169 and left_elbow_angle < 177 and right_elbow_angle > 199 and right_elbow_angle < 202 or \
        left_elbow_angle > 163 and left_elbow_angle < 172 and right_elbow_angle > 178 and right_elbow_angle < 189 or \
        left_elbow_angle > 163 and left_elbow_angle < 172 and right_elbow_angle > 192 and right_elbow_angle < 196 or \
        left_elbow_angle > 167 and left_elbow_angle < 172 and right_elbow_angle > 192 and right_elbow_angle < 196 or \
        left_elbow_angle > 241 and left_elbow_angle < 245 and right_elbow_angle > 11 and right_elbow_angle < 15 or \
        left_elbow_angle > 229 and left_elbow_angle < 233 and right_elbow_angle > 11 and right_elbow_angle < 15 or \
        left_elbow_angle > 266 and left_elbow_angle < 270 and right_elbow_angle > 342 and right_elbow_angle < 346 or \
        left_elbow_angle > 244 and left_elbow_angle < 250 and right_elbow_angle > 351 and right_elbow_angle < 358 or \
        left_elbow_angle > 270 and left_elbow_angle < 274 and right_elbow_angle > 351 and right_elbow_angle < 358 or \
        left_elbow_angle > 160 and left_elbow_angle < 164 and right_elbow_angle > 235 and right_elbow_angle < 239 or \
        left_elbow_angle > 181 and left_elbow_angle < 185 and right_elbow_angle > 207 and right_elbow_angle < 211 or \
        left_elbow_angle > 160 and left_elbow_angle < 175 and right_elbow_angle > 175 and right_elbow_angle < 185 or \
        left_elbow_angle > 160 and left_elbow_angle < 175 and right_elbow_angle > 190 and right_elbow_angle < 194 or \
        left_elbow_angle > 180 and left_elbow_angle < 186 and right_elbow_angle > 271 and right_elbow_angle < 288 or \
        left_elbow_angle > 4 and left_elbow_angle < 8 and right_elbow_angle > 301 and right_elbow_angle < 305  or \
        left_elbow_angle > 299 and left_elbow_angle < 303 and right_elbow_angle > 311 and right_elbow_angle < 315 or \
        left_elbow_angle > 256 and left_elbow_angle < 260 and right_elbow_angle > 326 and right_elbow_angle < 330 or \
        left_elbow_angle > 262 and left_elbow_angle < 266 and right_elbow_angle > 341 and right_elbow_angle < 345 or \
        left_elbow_angle > 242 and left_elbow_angle < 246 and right_elbow_angle > 337 and right_elbow_angle < 340 or \
        left_elbow_angle > 248 and left_elbow_angle < 252 and right_elbow_angle > 330 and right_elbow_angle < 334 or \
        left_elbow_angle > 297 and left_elbow_angle < 231 and right_elbow_angle > 331 and right_elbow_angle < 335 or \
        left_elbow_angle > 121 and left_elbow_angle < 125 and right_elbow_angle > 304 and right_elbow_angle < 308 or \
        left_elbow_angle > 156 and left_elbow_angle < 160 and right_elbow_angle > 273 and right_elbow_angle < 277 or \
        left_elbow_angle > 163 and left_elbow_angle < 169 and right_elbow_angle > 141 and right_elbow_angle < 245 or \
        left_elbow_angle > 165 and left_elbow_angle < 169 and right_elbow_angle > 198 and right_elbow_angle < 202 or \
        left_elbow_angle > 169 and left_elbow_angle < 183 and right_elbow_angle > 180 and right_elbow_angle < 187 or \
        left_elbow_angle > 169 and left_elbow_angle < 183 and right_elbow_angle > 223 and right_elbow_angle < 227 or \
        left_elbow_angle > 169 and left_elbow_angle < 183 and right_elbow_angle > 288 and right_elbow_angle < 292 or \
        left_elbow_angle > 350 and left_elbow_angle < 354 and right_elbow_angle > 305 and right_elbow_angle < 309 or \
        left_elbow_angle > 299 and left_elbow_angle < 303 and right_elbow_angle > 312 and right_elbow_angle < 316 or \
        left_elbow_angle > 241 and left_elbow_angle < 248 and right_elbow_angle > 331 and right_elbow_angle < 335 or \
        left_elbow_angle > 241 and left_elbow_angle < 248 and right_elbow_angle > 325 and right_elbow_angle < 329 or \
        left_elbow_angle > 241 and left_elbow_angle < 248 and right_elbow_angle > 332 and right_elbow_angle < 336 or \
        left_elbow_angle > 325 and left_elbow_angle < 239 and right_elbow_angle > 331 and right_elbow_angle < 335 or \
        left_elbow_angle > 255 and left_elbow_angle < 259 and right_elbow_angle > 229 and right_elbow_angle < 233 or \
        left_elbow_angle > 262 and left_elbow_angle < 266 and right_elbow_angle > 333 and right_elbow_angle < 237 or \
        left_elbow_angle > 128 and left_elbow_angle < 132 and right_elbow_angle > 311 and right_elbow_angle < 315 or \
        left_elbow_angle > 173 and left_elbow_angle < 177 and right_elbow_angle > 250 and right_elbow_angle < 254 or \
        left_elbow_angle > 161 and left_elbow_angle < 165 and right_elbow_angle > 231 and right_elbow_angle < 235 or \
        left_elbow_angle > 165 and left_elbow_angle < 169 and right_elbow_angle > 201 and right_elbow_angle < 205 or \
        left_elbow_angle > 170 and left_elbow_angle < 174 and right_elbow_angle > 187 and right_elbow_angle < 191 or \
        left_elbow_angle > 166 and left_elbow_angle < 177 and right_elbow_angle > 176 and right_elbow_angle < 190 or \
        left_elbow_angle > 110 and left_elbow_angle < 130 and right_elbow_angle > 280 and right_elbow_angle < 300:
        if left_shoulder_angle > 10 and left_shoulder_angle < 30 and right_shoulder_angle > 55 and right_shoulder_angle < 70 or \
            left_shoulder_angle > 10 and left_shoulder_angle < 30 and right_shoulder_angle > 25 and right_shoulder_angle < 60 or \
            left_shoulder_angle > 20 and left_shoulder_angle < 40 and right_shoulder_angle > 25 and right_shoulder_angle < 50 or \
            left_shoulder_angle > 20 and left_shoulder_angle < 40 and right_shoulder_angle > 60 and right_shoulder_angle < 90 or \
            left_shoulder_angle > 20 and left_shoulder_angle < 40 and right_shoulder_angle > 30 and right_shoulder_angle < 60 or \
            left_shoulder_angle > 40 and left_shoulder_angle < 60 and right_shoulder_angle > 125 and right_shoulder_angle < 145 or \
            left_shoulder_angle > 50 and left_shoulder_angle < 70 and right_shoulder_angle > 120 and right_shoulder_angle < 150 or \
            left_shoulder_angle > 120 and left_shoulder_angle < 140 and right_shoulder_angle > 50 and right_shoulder_angle < 70 or \
            left_shoulder_angle > 305 and left_shoulder_angle < 330 and right_shoulder_angle > 15 and right_shoulder_angle < 50 or \
            left_shoulder_angle > 15 and left_shoulder_angle < 21 and right_shoulder_angle > 20 and right_shoulder_angle < 40 or \
            left_shoulder_angle > 0 and left_shoulder_angle < 3 and right_shoulder_angle > 22 and right_shoulder_angle < 26 or\
            left_shoulder_angle > 330 and left_shoulder_angle < 336 and right_shoulder_angle > 14 and right_shoulder_angle < 18 or\
            left_shoulder_angle > 340 and left_shoulder_angle < 346 and right_shoulder_angle > 8 and right_shoulder_angle < 12 or\
            left_shoulder_angle > 340 and left_shoulder_angle < 346 and right_shoulder_angle > 8 and right_shoulder_angle < 16 or\
            left_shoulder_angle > 46 and left_shoulder_angle < 50 and right_shoulder_angle > 57 and right_shoulder_angle < 61 or\
            left_shoulder_angle > 35 and left_shoulder_angle < 39 and right_shoulder_angle > 90 and right_shoulder_angle < 94 or\
            left_shoulder_angle > 23 and left_shoulder_angle < 27 and right_shoulder_angle > 115 and right_shoulder_angle < 118 or\
            left_shoulder_angle > 14 and left_shoulder_angle < 22 and right_shoulder_angle > 100 and right_shoulder_angle < 122 or\
            left_shoulder_angle > 14 and left_shoulder_angle < 22 and right_shoulder_angle > 100 and right_shoulder_angle < 122 or\
            left_shoulder_angle > 14 and left_shoulder_angle < 22 and right_shoulder_angle > 130 and right_shoulder_angle < 134 or\
            left_shoulder_angle > 14 and left_shoulder_angle < 22 and right_shoulder_angle > 94 and right_shoulder_angle < 96 or\
            left_shoulder_angle > 13 and left_shoulder_angle < 21 and right_shoulder_angle > 94 and right_shoulder_angle < 96 or\
            left_shoulder_angle > 347 and left_shoulder_angle < 360 and right_shoulder_angle > 8 and right_shoulder_angle < 13 or\
            left_shoulder_angle > 25 and left_shoulder_angle < 29 and right_shoulder_angle > 81 and right_shoulder_angle < 85 or\
            left_shoulder_angle > 11 and left_shoulder_angle < 19 and right_shoulder_angle > 120 and right_shoulder_angle < 136 or\
            left_shoulder_angle > 21 and left_shoulder_angle < 27 and right_shoulder_angle > 54 and right_shoulder_angle < 59 or \
            left_shoulder_angle > 289 and left_shoulder_angle < 292 and right_shoulder_angle > 36 and right_shoulder_angle < 40 or\
            left_shoulder_angle > 323 and left_shoulder_angle < 327 and right_shoulder_angle > 30 and right_shoulder_angle < 34 or\
            left_shoulder_angle > 326 and left_shoulder_angle < 330 and right_shoulder_angle > 23 and right_shoulder_angle < 27 or\
            left_shoulder_angle > 6 and left_shoulder_angle < 10 and right_shoulder_angle > 28 and right_shoulder_angle < 32 or\
            left_shoulder_angle > 65 and left_shoulder_angle < 69 and right_shoulder_angle > 90 and right_shoulder_angle < 94 or\
            left_shoulder_angle > 51 and left_shoulder_angle < 55 and right_shoulder_angle > 117 and right_shoulder_angle < 121 or\
            left_shoulder_angle > 38 and left_shoulder_angle < 42 and right_shoulder_angle > 149 and right_shoulder_angle < 157 or\
            left_shoulder_angle > 30 and left_shoulder_angle < 34 and right_shoulder_angle > 149 and right_shoulder_angle < 157 or\
            left_shoulder_angle > 29 and left_shoulder_angle < 41 and right_shoulder_angle > 149 and right_shoulder_angle < 155 or\
            left_shoulder_angle > 29 and left_shoulder_angle < 41 and right_shoulder_angle > 143 and right_shoulder_angle < 147 or\
            left_shoulder_angle > 29 and left_shoulder_angle < 41 and right_shoulder_angle > 131 and right_shoulder_angle < 135 or\
            left_shoulder_angle > 46 and left_shoulder_angle < 50 and right_shoulder_angle > 23 and right_shoulder_angle < 27 or\
            left_shoulder_angle > 75 and left_shoulder_angle < 79 and right_shoulder_angle > 15 and right_shoulder_angle < 31 or\
            left_shoulder_angle > 87 and left_shoulder_angle < 89 and right_shoulder_angle > 15 and right_shoulder_angle < 31 or\
            left_shoulder_angle > 18 and left_shoulder_angle < 22 and right_shoulder_angle > 15 and right_shoulder_angle < 31 or\
            left_shoulder_angle > 346 and left_shoulder_angle < 350 and right_shoulder_angle > 15 and right_shoulder_angle < 31 or\
            left_shoulder_angle > 337 and left_shoulder_angle < 352 and right_shoulder_angle > 6 and right_shoulder_angle < 22 or\
            left_shoulder_angle > 58 and left_shoulder_angle < 62 and right_shoulder_angle > 106 and right_shoulder_angle < 110 or\
            left_shoulder_angle > 41 and left_shoulder_angle < 51 and right_shoulder_angle > 129 and right_shoulder_angle < 133 or\
            left_shoulder_angle > 41 and left_shoulder_angle < 51 and right_shoulder_angle > 142 and right_shoulder_angle < 146 or\
            left_shoulder_angle > 41 and left_shoulder_angle < 51 and right_shoulder_angle > 137 and right_shoulder_angle < 141 or\
            left_shoulder_angle > 23 and left_shoulder_angle < 28 and right_shoulder_angle > 132 and right_shoulder_angle < 147 or\
            left_shoulder_angle > 322 and left_shoulder_angle < 345 and right_shoulder_angle > 9 and right_shoulder_angle < 21 or\
            left_shoulder_angle > 57 and left_shoulder_angle < 62 and right_shoulder_angle > 105 and right_shoulder_angle < 109 or\
            left_shoulder_angle > 57 and left_shoulder_angle < 62 and right_shoulder_angle > 126 and right_shoulder_angle < 130 or\
            left_shoulder_angle > 36 and left_shoulder_angle < 40 and right_shoulder_angle > 142 and right_shoulder_angle < 148 or\
            left_shoulder_angle > 22 and left_shoulder_angle < 30 and right_shoulder_angle > 140 and right_shoulder_angle < 148 or\
            left_shoulder_angle > 22 and left_shoulder_angle < 30 and right_shoulder_angle > 132 and right_shoulder_angle < 136 or\
            left_shoulder_angle > 16 and left_shoulder_angle < 22 and right_shoulder_angle > 46 and right_shoulder_angle < 54 or \
            left_shoulder_angle > 18 and left_shoulder_angle < 22 and right_shoulder_angle > 41 and right_shoulder_angle < 45 or\
            left_shoulder_angle > 28 and left_shoulder_angle < 32 and right_shoulder_angle > 34 and right_shoulder_angle < 38 or\
            left_shoulder_angle > 357 and left_shoulder_angle < 360 and right_shoulder_angle > 29 and right_shoulder_angle < 33 or\
            left_shoulder_angle > 327 and left_shoulder_angle < 331 and right_shoulder_angle > 26 and right_shoulder_angle < 33 or\
            left_shoulder_angle > 340 and left_shoulder_angle < 344 and right_shoulder_angle > 18 and right_shoulder_angle < 22 or\
            left_shoulder_angle > 325 and left_shoulder_angle < 329 and right_shoulder_angle > 20 and right_shoulder_angle < 26 or\
            left_shoulder_angle > 332 and left_shoulder_angle < 336 and right_shoulder_angle > 22 and right_shoulder_angle < 26 or\
            left_shoulder_angle > 3 and left_shoulder_angle < 7 and right_shoulder_angle > 21 and right_shoulder_angle < 25 or\
            left_shoulder_angle > 25 and left_shoulder_angle < 29 and right_shoulder_angle > 98 and right_shoulder_angle < 102 or\
            left_shoulder_angle > 40 and left_shoulder_angle < 44 and right_shoulder_angle > 103 and right_shoulder_angle < 107  or\
            left_shoulder_angle > 35 and left_shoulder_angle < 39 and right_shoulder_angle > 126 and right_shoulder_angle < 130 or\
            left_shoulder_angle > 24 and left_shoulder_angle < 28 and right_shoulder_angle > 150 and right_shoulder_angle < 154 or\
            left_shoulder_angle > 16 and left_shoulder_angle < 22 and right_shoulder_angle > 157 and right_shoulder_angle < 162 or\
            left_shoulder_angle > 16 and left_shoulder_angle < 20 and right_shoulder_angle > 154 and right_shoulder_angle < 158 or\
            left_shoulder_angle > 18 and left_shoulder_angle < 25 and right_shoulder_angle > 149 and right_shoulder_angle < 153 or\
            left_shoulder_angle > 18 and left_shoulder_angle < 25 and right_shoulder_angle > 114 and right_shoulder_angle < 118 or\
            left_shoulder_angle > 18 and left_shoulder_angle < 25 and right_shoulder_angle > 40 and right_shoulder_angle < 44 or\
            left_shoulder_angle > 16 and left_shoulder_angle < 20 and right_shoulder_angle > 28 and right_shoulder_angle < 33 or\
            left_shoulder_angle > 349 and left_shoulder_angle < 353 and right_shoulder_angle > 26 and right_shoulder_angle < 30 or\
            left_shoulder_angle > 316 and left_shoulder_angle < 320 and right_shoulder_angle > 24 and right_shoulder_angle < 30 or\
            left_shoulder_angle > 325 and left_shoulder_angle < 329 and right_shoulder_angle > 17 and right_shoulder_angle < 21 or\
            left_shoulder_angle > 313 and left_shoulder_angle < 317 and right_shoulder_angle > 16 and right_shoulder_angle < 20 or\
            left_shoulder_angle > 330 and left_shoulder_angle < 334 and right_shoulder_angle > 19 and right_shoulder_angle < 23 or\
            left_shoulder_angle > 340 and left_shoulder_angle < 344 and right_shoulder_angle > 21 and right_shoulder_angle < 25 or\
            left_shoulder_angle > 48 and left_shoulder_angle < 52 and right_shoulder_angle > 20 and right_shoulder_angle < 24 or\
            left_shoulder_angle > 41 and left_shoulder_angle < 45 and right_shoulder_angle > 105 and right_shoulder_angle < 109 or\
            left_shoulder_angle > 28 and left_shoulder_angle < 32 and right_shoulder_angle > 129 and right_shoulder_angle < 132 or\
            left_shoulder_angle > 14 and left_shoulder_angle < 19 and right_shoulder_angle > 158 and right_shoulder_angle < 163 or\
            left_shoulder_angle > 13 and left_shoulder_angle < 21 and right_shoulder_angle > 155 and right_shoulder_angle < 164 or\
            left_shoulder_angle > 13 and left_shoulder_angle < 21 and right_shoulder_angle > 141 and right_shoulder_angle < 145 or\
            left_shoulder_angle > 13 and left_shoulder_angle < 21 and right_shoulder_angle > 121 and right_shoulder_angle < 125 or\
            left_shoulder_angle > 320 and left_shoulder_angle < 350 and right_shoulder_angle > 15 and right_shoulder_angle < 50 :
            if left_knee_angle > 165 and left_knee_angle < 180 and right_knee_angle > 170 and right_knee_angle < 195 or \
                left_knee_angle > 160 and left_knee_angle < 190 and right_knee_angle > 90 and right_knee_angle < 130 or \
                left_knee_angle > 165 and left_knee_angle < 190 and right_knee_angle > 140 and right_knee_angle < 150 or \
                left_knee_angle > 230 and left_knee_angle < 260 and right_knee_angle > 155 and right_knee_angle < 175 or \
                 left_knee_angle > 178 and left_knee_angle < 179 and right_knee_angle > 179 and right_knee_angle < 188 or \
                left_knee_angle > 197 and left_knee_angle < 201 and right_knee_angle > 177 and right_knee_angle < 185 or \
                left_knee_angle > 208 and left_knee_angle < 224 and right_knee_angle > 179 and right_knee_angle < 182 or \
                left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 187 and right_knee_angle < 191 or \
                left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 151 and right_knee_angle < 155 or \
                left_knee_angle > 164 and left_knee_angle < 168 and right_knee_angle > 113 and right_knee_angle < 117 or \
                left_knee_angle > 152 and left_knee_angle < 161 and right_knee_angle > 116 and right_knee_angle < 125 or \
                left_knee_angle > 160 and left_knee_angle < 168 and right_knee_angle > 122 and right_knee_angle < 128 or \
                left_knee_angle > 160 and left_knee_angle < 168 and right_knee_angle > 133 and right_knee_angle < 137 or \
                left_knee_angle > 170 and left_knee_angle < 177 and right_knee_angle > 176 and right_knee_angle < 187 or \
                left_knee_angle > 204 and left_knee_angle < 209 and right_knee_angle > 172 and right_knee_angle < 181 or \
                left_knee_angle > 170 and left_knee_angle < 174 and right_knee_angle > 129 and right_knee_angle < 133 or \
                left_knee_angle > 152 and left_knee_angle < 168 and right_knee_angle > 121 and right_knee_angle < 131 or \
                left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 176 and right_knee_angle < 180 or \
                left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 179 and right_knee_angle < 183 or \
                left_knee_angle > 237 and left_knee_angle < 249 and right_knee_angle > 263 and right_knee_angle < 168 or \
                left_knee_angle > 226 and left_knee_angle < 232 and right_knee_angle > 171 and right_knee_angle < 175 or \
                left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 144 and right_knee_angle < 148 or \
                left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 127 and right_knee_angle < 131 or \
                left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 131 and right_knee_angle < 135 or \
                left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 143 and right_knee_angle < 147 or \
                left_knee_angle > 179 and left_knee_angle < 187 and right_knee_angle > 135 and right_knee_angle < 139 or \
                left_knee_angle > 179 and left_knee_angle < 187 and right_knee_angle > 140 and right_knee_angle < 144 or \
                left_knee_angle > 179 and left_knee_angle < 187 and right_knee_angle > 129 and right_knee_angle < 133 or \
                left_knee_angle > 173 and left_knee_angle < 179 and right_knee_angle > 131 and right_knee_angle < 135 or \
                left_knee_angle > 173 and left_knee_angle < 179 and right_knee_angle > 122 and right_knee_angle < 126 or \
                left_knee_angle > 173 and left_knee_angle < 179 and right_knee_angle > 170 and right_knee_angle < 176 or \
                left_knee_angle > 234 and left_knee_angle < 238 and right_knee_angle > 165 and right_knee_angle < 173 or \
                left_knee_angle > 242 and left_knee_angle < 248 and right_knee_angle > 165 and right_knee_angle < 173 or \
                left_knee_angle > 234 and left_knee_angle < 241 and right_knee_angle > 164 and right_knee_angle < 172 or \
                left_knee_angle > 220 and left_knee_angle < 231 and right_knee_angle > 163 and right_knee_angle < 169 or \
                left_knee_angle > 220 and left_knee_angle < 231 and right_knee_angle > 170 and right_knee_angle < 180 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 173 and right_knee_angle < 177 or \
                left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 122 and right_knee_angle < 126 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 131 and right_knee_angle < 135 or \
                left_knee_angle > 149 and left_knee_angle < 153 and right_knee_angle > 122 and right_knee_angle < 126 or \
                left_knee_angle > 120 and left_knee_angle < 124 and right_knee_angle > 128 and right_knee_angle < 133 or \
                left_knee_angle > 134 and left_knee_angle < 138 and right_knee_angle > 128 and right_knee_angle < 133 or \
                left_knee_angle > 120 and left_knee_angle < 124 and right_knee_angle > 128 and right_knee_angle < 132 or \
                left_knee_angle > 134 and left_knee_angle < 138 and right_knee_angle > 128 and right_knee_angle < 132 or \
                left_knee_angle > 160 and left_knee_angle < 164 and right_knee_angle > 133 and right_knee_angle < 137 or \
                left_knee_angle > 157 and left_knee_angle < 162 and right_knee_angle > 134 and right_knee_angle < 138 or \
                left_knee_angle > 183 and left_knee_angle < 187 and right_knee_angle > 146 and right_knee_angle < 150 or \
                left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle >140 and right_knee_angle < 144 or \
                left_knee_angle > 210 and left_knee_angle < 214 and right_knee_angle > 164 and right_knee_angle < 168 or \
                left_knee_angle > 235 and left_knee_angle < 239 and right_knee_angle > 166 and right_knee_angle < 170 or \
                left_knee_angle > 220 and left_knee_angle < 231 and right_knee_angle > 168 and right_knee_angle < 177 or \
                left_knee_angle > 220 and left_knee_angle < 231 and right_knee_angle > 161 and right_knee_angle < 165 or \
                left_knee_angle > 169 and left_knee_angle < 184 and right_knee_angle > 161 and right_knee_angle < 165 or \
                left_knee_angle > 169 and left_knee_angle < 184 and right_knee_angle > 131 and right_knee_angle < 135 or \
                left_knee_angle > 169 and left_knee_angle < 184 and right_knee_angle > 119 and right_knee_angle < 133 or \
                left_knee_angle > 169 and left_knee_angle < 184 and right_knee_angle > 141 and right_knee_angle < 145 or \
                left_knee_angle > 168 and left_knee_angle < 178 and right_knee_angle > 179 and right_knee_angle < 188 or \
                left_knee_angle > 210 and left_knee_angle < 214 and right_knee_angle > 167 and right_knee_angle < 171 or \
                left_knee_angle > 225 and left_knee_angle < 235 and right_knee_angle > 163 and right_knee_angle < 170 or \
                left_knee_angle > 217 and left_knee_angle < 221 and right_knee_angle > 166 and right_knee_angle < 170 or \
                left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 178 and right_knee_angle < 182 or \
                left_knee_angle > 170 and left_knee_angle < 177 and right_knee_angle > 175 and right_knee_angle < 179 or \
                left_knee_angle > 170 and left_knee_angle < 177 and right_knee_angle > 145 and right_knee_angle < 149 or \
                left_knee_angle > 157 and left_knee_angle < 169 and right_knee_angle > 111 and right_knee_angle < 123 or \
                left_knee_angle > 157 and left_knee_angle < 169 and right_knee_angle > 133 and right_knee_angle < 138 or \
                left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 179 and right_knee_angle < 183 or \
                left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 172 and right_knee_angle < 176 or \
                left_knee_angle > 193 and left_knee_angle < 197 and right_knee_angle > 169 and right_knee_angle < 173 or \
                left_knee_angle > 208 and left_knee_angle < 212 and right_knee_angle > 163 and right_knee_angle < 167 or \
                left_knee_angle > 219 and left_knee_angle < 226 and right_knee_angle > 163 and right_knee_angle < 177 or \
                left_knee_angle > 211 and left_knee_angle < 216 and right_knee_angle > 163 and right_knee_angle < 177 or \
                left_knee_angle > 171 and left_knee_angle < 181 and right_knee_angle > 134 and right_knee_angle < 138 or \
                left_knee_angle > 171 and left_knee_angle < 181 and right_knee_angle > 111 and right_knee_angle < 115 or \
                left_knee_angle > 171 and left_knee_angle < 181 and right_knee_angle > 130 and right_knee_angle < 134 or \
                left_knee_angle > 164 and left_knee_angle < 168 and right_knee_angle > 126 and right_knee_angle < 130 or \
                left_knee_angle > 166 and left_knee_angle < 170 and right_knee_angle > 131 and right_knee_angle < 135 or \
                left_knee_angle > 175 and left_knee_angle < 182 and right_knee_angle > 132 and right_knee_angle < 144 or \
                left_knee_angle > 164 and left_knee_angle < 170 and right_knee_angle > 113 and right_knee_angle < 117 or \
                left_knee_angle > 164 and left_knee_angle < 170 and right_knee_angle > 132 and right_knee_angle < 136 or \
                left_knee_angle > 240 and left_knee_angle < 260 and right_knee_angle > 160 and right_knee_angle < 180 :
                label = "Tepuk"    
                last_label = label 
    
        if left_elbow_angle > 120 and left_elbow_angle < 140 and right_elbow_angle > 250 and right_elbow_angle < 280 or \
            left_elbow_angle > 170 and left_elbow_angle < 200 and right_elbow_angle > 160 and right_elbow_angle < 190 or \
            left_elbow_angle > 340 and left_elbow_angle < 360 and right_elbow_angle > 20 and right_elbow_angle < 40 or \
            left_elbow_angle > 320 and left_elbow_angle < 360 and right_elbow_angle > 0 and right_elbow_angle < 30 or \
            left_elbow_angle > 300 and left_elbow_angle < 360 and right_elbow_angle > 330 and right_elbow_angle < 360 or \
            left_elbow_angle > 340 and left_elbow_angle < 360 and right_elbow_angle > 20 and right_elbow_angle < 50 or \
            left_elbow_angle > 300 and left_elbow_angle < 330 and right_elbow_angle > 20 and right_elbow_angle < 60 or \
            left_elbow_angle > 20 and left_elbow_angle < 50 and right_elbow_angle > 30 and right_elbow_angle < 60 or \
            left_elbow_angle > 0 and left_elbow_angle < 30 and right_elbow_angle > 30 and right_elbow_angle < 60 or \
            left_elbow_angle > 0 and left_elbow_angle < 20 and right_elbow_angle > 00 and right_elbow_angle < 20 or \
            left_elbow_angle > 158 and left_elbow_angle < 162 and right_elbow_angle > 178 and right_elbow_angle < 182 or \
            left_elbow_angle > 158 and left_elbow_angle < 162 and right_elbow_angle > 178 and right_elbow_angle < 182 or \
            left_elbow_angle > 160 and left_elbow_angle < 164 and right_elbow_angle > 176 and right_elbow_angle < 180 or \
            left_elbow_angle > 155 and left_elbow_angle < 159 and right_elbow_angle > 282 and right_elbow_angle < 286 or \
            left_elbow_angle > 144 and left_elbow_angle < 148 and right_elbow_angle > 182 and right_elbow_angle < 186 or \
            left_elbow_angle > 50  and left_elbow_angle < 54  and right_elbow_angle > 122 and right_elbow_angle < 126 or \
            left_elbow_angle > 232 and left_elbow_angle < 236 and right_elbow_angle > 137 and right_elbow_angle < 141 or \
            left_elbow_angle > 172 and left_elbow_angle < 176 and right_elbow_angle > 117 and right_elbow_angle < 121 or \
            left_elbow_angle > 77  and left_elbow_angle < 81  and right_elbow_angle > 55  and right_elbow_angle < 59  or \
            left_elbow_angle > 0   and left_elbow_angle < 2   and right_elbow_angle > 18  and right_elbow_angle < 22  or \
            left_elbow_angle > 354 and left_elbow_angle < 358 and right_elbow_angle > 68  and right_elbow_angle < 72  or \
            left_elbow_angle > 119 and left_elbow_angle < 123 and right_elbow_angle > 122 and right_elbow_angle < 126 or \
            left_elbow_angle > 163 and left_elbow_angle < 167 and right_elbow_angle > 144 and right_elbow_angle < 148 or \
            left_elbow_angle > 176 and left_elbow_angle < 180 and right_elbow_angle > 163 and right_elbow_angle < 167 or \
            left_elbow_angle > 183 and left_elbow_angle < 187 and right_elbow_angle > 152 and right_elbow_angle < 156 or \
            left_elbow_angle > 175 and left_elbow_angle < 179 and right_elbow_angle > 143 and right_elbow_angle < 147 or \
            left_elbow_angle > 171 and left_elbow_angle < 175 and right_elbow_angle > 141 and right_elbow_angle < 145 or \
            left_elbow_angle > 177 and left_elbow_angle < 181 and right_elbow_angle > 140 and right_elbow_angle < 144 or \
            left_elbow_angle > 181 and left_elbow_angle < 185 and right_elbow_angle > 144 and right_elbow_angle < 148 or \
            left_elbow_angle > 74  and left_elbow_angle < 78  and right_elbow_angle > 338 and right_elbow_angle < 342 or \
            left_elbow_angle > 212 and left_elbow_angle < 216 and right_elbow_angle > 156 and right_elbow_angle < 160 or \
            left_elbow_angle > 239 and left_elbow_angle < 243 and right_elbow_angle > 122 and right_elbow_angle < 126 or \
            left_elbow_angle > 240 and left_elbow_angle < 244 and right_elbow_angle > 97  and right_elbow_angle < 101 or \
            left_elbow_angle > 240 and left_elbow_angle < 244 and right_elbow_angle > 114 and right_elbow_angle < 118 or \
            left_elbow_angle > 255 and left_elbow_angle < 259 and right_elbow_angle > 180 and right_elbow_angle < 184 or \
            left_elbow_angle > 232 and left_elbow_angle < 236 and right_elbow_angle > 157 and right_elbow_angle < 161 or \
            left_elbow_angle > 264 and left_elbow_angle < 268 and right_elbow_angle > 74  and right_elbow_angle < 78  or \
            left_elbow_angle > 0   and left_elbow_angle < 4   and right_elbow_angle > 354 and right_elbow_angle < 358 or \
            left_elbow_angle > 0   and left_elbow_angle < 3   and right_elbow_angle > 356 and right_elbow_angle < 360 or \
            left_elbow_angle > 0   and left_elbow_angle < 3   and right_elbow_angle > 355 and right_elbow_angle < 360 or \
            left_elbow_angle > 273 and left_elbow_angle < 277 and right_elbow_angle > 83  and right_elbow_angle < 87  or \
            left_elbow_angle > 203 and left_elbow_angle < 207 and right_elbow_angle > 164 and right_elbow_angle < 168 or \
            left_elbow_angle > 185 and left_elbow_angle < 189 and right_elbow_angle > 155 and right_elbow_angle < 159 or \
            left_elbow_angle > 136 and left_elbow_angle < 140 and right_elbow_angle > 150 and right_elbow_angle < 154 or \
            left_elbow_angle > 70  and left_elbow_angle < 74  and right_elbow_angle > 121 and right_elbow_angle < 125 or \
            left_elbow_angle > 241 and left_elbow_angle < 245 and right_elbow_angle > 129 and right_elbow_angle < 133 or \
            left_elbow_angle > 108 and left_elbow_angle < 112 and right_elbow_angle > 124 and right_elbow_angle < 128 or \
            left_elbow_angle > 144 and left_elbow_angle < 148 and right_elbow_angle > 97  and right_elbow_angle < 101 or \
            left_elbow_angle > 2   and left_elbow_angle < 6   and right_elbow_angle > 357 and right_elbow_angle < 360 or \
            left_elbow_angle > 0   and left_elbow_angle < 4   and right_elbow_angle > 357 and right_elbow_angle < 360 or \
            left_elbow_angle > 355 and left_elbow_angle < 359 and right_elbow_angle > 69  and right_elbow_angle < 73  or \
            left_elbow_angle > 96  and left_elbow_angle < 100 and right_elbow_angle > 106 and right_elbow_angle < 110 or \
            left_elbow_angle > 174 and left_elbow_angle < 178 and right_elbow_angle > 142 and right_elbow_angle < 146 or \
            left_elbow_angle > 173 and left_elbow_angle < 177 and right_elbow_angle > 140 and right_elbow_angle < 144 or \
            left_elbow_angle > 144 and left_elbow_angle < 148 and right_elbow_angle > 116 and right_elbow_angle < 120 or \
            left_elbow_angle > 159 and left_elbow_angle < 163 and right_elbow_angle > 114 and right_elbow_angle < 118 or \
            left_elbow_angle > 78 and left_elbow_angle < 82 and right_elbow_angle > 253 and right_elbow_angle < 257 or \
            left_elbow_angle > 11 and left_elbow_angle < 15 and right_elbow_angle > 310 and right_elbow_angle < 314 or \
            left_elbow_angle > 353 and left_elbow_angle < 357 and right_elbow_angle > 321 and right_elbow_angle < 325 or \
            left_elbow_angle > 362 and left_elbow_angle < 366 and right_elbow_angle > 30  and right_elbow_angle < 34  or \
            left_elbow_angle > 269 and left_elbow_angle < 273 and right_elbow_angle > 28  and right_elbow_angle < 32  or \
            left_elbow_angle > 264 and left_elbow_angle < 268 and right_elbow_angle > 53  and right_elbow_angle < 57  or \
            left_elbow_angle > 258 and left_elbow_angle < 262 and right_elbow_angle > 45  and right_elbow_angle < 49  or \
            left_elbow_angle > 311 and left_elbow_angle < 315 and right_elbow_angle > 356 and right_elbow_angle < 360 or \
            left_elbow_angle > 332 and left_elbow_angle < 336 and right_elbow_angle > 8   and right_elbow_angle < 12  or \
            left_elbow_angle > 314 and left_elbow_angle < 318 and right_elbow_angle > 10  and right_elbow_angle < 14  or \
            left_elbow_angle > 276 and left_elbow_angle < 280 and right_elbow_angle > 37  and right_elbow_angle < 41  or \
            left_elbow_angle > 270 and left_elbow_angle < 274 and right_elbow_angle > 60  and right_elbow_angle < 64  or \
            left_elbow_angle > 274 and left_elbow_angle < 278 and right_elbow_angle > 55  and right_elbow_angle < 59  or \
            left_elbow_angle > 281 and left_elbow_angle < 285 and right_elbow_angle > 28  and right_elbow_angle < 32  or \
            left_elbow_angle > 272 and left_elbow_angle < 276 and right_elbow_angle > 44  and right_elbow_angle < 48  or \
            left_elbow_angle > 271 and left_elbow_angle < 275 and right_elbow_angle > 17  and right_elbow_angle < 21  or \
            left_elbow_angle > 280 and left_elbow_angle < 284 and right_elbow_angle > 44  and right_elbow_angle < 48  or \
            left_elbow_angle > 275 and left_elbow_angle < 279 and right_elbow_angle > 21  and right_elbow_angle < 25  or \
            left_elbow_angle > 291 and left_elbow_angle < 295 and right_elbow_angle > 30  and right_elbow_angle < 34  or \
            left_elbow_angle > 270 and left_elbow_angle < 274 and right_elbow_angle > 120 and right_elbow_angle < 124 or \
            left_elbow_angle > 211 and left_elbow_angle < 215 and right_elbow_angle > 172 and right_elbow_angle < 176 or \
            left_elbow_angle > 179 and left_elbow_angle < 183 and right_elbow_angle > 297 and right_elbow_angle < 301 or \
            left_elbow_angle > 43  and left_elbow_angle < 47  and right_elbow_angle > 354 and right_elbow_angle < 358 or \
            left_elbow_angle > 334 and left_elbow_angle < 338 and right_elbow_angle > 45  and right_elbow_angle < 49  or \
            left_elbow_angle > 296 and left_elbow_angle < 290 and right_elbow_angle > 75  and right_elbow_angle < 79  or \
            left_elbow_angle > 329 and left_elbow_angle < 333 and right_elbow_angle > 29  and right_elbow_angle < 33  or \
            left_elbow_angle > 311 and left_elbow_angle < 315 and right_elbow_angle > 60  and right_elbow_angle < 64  or \
            left_elbow_angle > 308 and left_elbow_angle < 312 and right_elbow_angle > 64  and right_elbow_angle < 68  or \
            left_elbow_angle > 295 and left_elbow_angle < 299 and right_elbow_angle > 63  and right_elbow_angle < 67  or \
            left_elbow_angle > 288 and left_elbow_angle < 292 and right_elbow_angle > 63  and right_elbow_angle < 67  or \
            left_elbow_angle > 245 and left_elbow_angle < 249 and right_elbow_angle > 84  and right_elbow_angle < 88  or \
            left_elbow_angle > 74  and left_elbow_angle < 78  and right_elbow_angle > 260 and right_elbow_angle < 264 or \
            left_elbow_angle > 2   and left_elbow_angle < 6   and right_elbow_angle > 319 and right_elbow_angle < 323 or \
            left_elbow_angle > 309 and left_elbow_angle < 313 and right_elbow_angle > 9   and right_elbow_angle < 13  or \
            left_elbow_angle > 296 and left_elbow_angle < 300 and right_elbow_angle > 26  and right_elbow_angle < 30  or \
            left_elbow_angle > 276 and left_elbow_angle < 280 and right_elbow_angle > 57  and right_elbow_angle < 61  or \
            left_elbow_angle > 275 and left_elbow_angle < 279 and right_elbow_angle > 43  and right_elbow_angle < 47  or \
            left_elbow_angle > 279 and left_elbow_angle < 283 and right_elbow_angle > 25  and right_elbow_angle < 29  or \
            left_elbow_angle > 310 and left_elbow_angle < 314 and right_elbow_angle > 13  and right_elbow_angle < 17  or \
            left_elbow_angle > 311 and left_elbow_angle < 315 and right_elbow_angle > 7   and right_elbow_angle < 11  or \
            left_elbow_angle > 295 and left_elbow_angle < 299 and right_elbow_angle > 11  and right_elbow_angle < 15  or \
            left_elbow_angle > 295 and left_elbow_angle < 299 and right_elbow_angle > 26  and right_elbow_angle < 30  or \
            left_elbow_angle > 285 and left_elbow_angle < 289 and right_elbow_angle > 23  and right_elbow_angle < 27  or \
            left_elbow_angle > 278 and left_elbow_angle < 282 and right_elbow_angle > 43  and right_elbow_angle < 47  or \
            left_elbow_angle > 285 and left_elbow_angle < 289 and right_elbow_angle > 12  and right_elbow_angle < 16  or \
            left_elbow_angle > 274 and left_elbow_angle < 278 and right_elbow_angle > 25  and right_elbow_angle < 29  or \
            left_elbow_angle > 263 and left_elbow_angle < 267 and right_elbow_angle > 48  and right_elbow_angle < 52  or \
            left_elbow_angle > 273 and left_elbow_angle < 277 and right_elbow_angle > 40  and right_elbow_angle < 44  or \
            left_elbow_angle > 274 and left_elbow_angle < 278 and right_elbow_angle > 42  and right_elbow_angle < 46  or \
            left_elbow_angle > 281 and left_elbow_angle < 285 and right_elbow_angle > 53  and right_elbow_angle < 57  or \
            left_elbow_angle > 270 and left_elbow_angle < 274 and right_elbow_angle > 57  and right_elbow_angle < 61  or \
            left_elbow_angle > 283 and left_elbow_angle < 287 and right_elbow_angle > 50  and right_elbow_angle < 54  or \
            left_elbow_angle > 275 and left_elbow_angle < 279 and right_elbow_angle > 57  and right_elbow_angle < 61  or \
            left_elbow_angle > 254 and left_elbow_angle < 258 and right_elbow_angle > 164 and right_elbow_angle < 168 or \
            left_elbow_angle > 172 and left_elbow_angle < 176 and right_elbow_angle > 181 and right_elbow_angle < 185 or \
            left_elbow_angle > 165 and left_elbow_angle < 169 and right_elbow_angle > 245 and right_elbow_angle < 249 or \
            left_elbow_angle > 145 and left_elbow_angle < 149 and right_elbow_angle > 265 and right_elbow_angle < 269 or \
            left_elbow_angle > 131 and left_elbow_angle < 135 and right_elbow_angle > 305 and right_elbow_angle < 309 or \
            left_elbow_angle > 115 and left_elbow_angle < 119 and right_elbow_angle > 53  and right_elbow_angle < 57  or \
            left_elbow_angle > 50  and left_elbow_angle < 54  and right_elbow_angle > 108 and right_elbow_angle < 112 or \
            left_elbow_angle > 31  and left_elbow_angle < 35  and right_elbow_angle > 91  and right_elbow_angle < 95  or \
            left_elbow_angle > 320 and left_elbow_angle < 324 and right_elbow_angle > 140 and right_elbow_angle < 144 or \
            left_elbow_angle > 308 and left_elbow_angle < 312 and right_elbow_angle > 129 and right_elbow_angle < 133 or \
            left_elbow_angle > 354 and left_elbow_angle < 358 and right_elbow_angle > 85  and right_elbow_angle < 89  or \
            left_elbow_angle > 3   and left_elbow_angle < 7   and right_elbow_angle > 351 and right_elbow_angle < 355 or \
            left_elbow_angle > 13  and left_elbow_angle < 17  and right_elbow_angle > 63  and right_elbow_angle < 67  or \
            left_elbow_angle > 68  and left_elbow_angle < 72  and right_elbow_angle > 121 and right_elbow_angle < 125 or \
            left_elbow_angle > 114 and left_elbow_angle < 118 and right_elbow_angle > 148 and right_elbow_angle < 152 or \
            left_elbow_angle > 147 and left_elbow_angle < 151 and right_elbow_angle > 135 and right_elbow_angle < 139 or \
            left_elbow_angle > 343 and left_elbow_angle < 357 and right_elbow_angle > 115 and right_elbow_angle < 119 or \
            left_elbow_angle > 296 and left_elbow_angle < 300 and right_elbow_angle > 131 and right_elbow_angle < 135 or \
            left_elbow_angle > 352 and left_elbow_angle < 356 and right_elbow_angle > 137 and right_elbow_angle < 141 or \
            left_elbow_angle > 291 and left_elbow_angle < 295 and right_elbow_angle > 130 and right_elbow_angle < 134 or \
            left_elbow_angle > 274 and left_elbow_angle < 278 and right_elbow_angle > 149 and right_elbow_angle < 153 or \
            left_elbow_angle > 153 and left_elbow_angle < 157 and right_elbow_angle > 197 and right_elbow_angle < 201 or \
            left_elbow_angle > 3   and left_elbow_angle < 7   and right_elbow_angle > 344 and right_elbow_angle < 348 or \
            left_elbow_angle > 293 and left_elbow_angle < 297 and right_elbow_angle > 53  and right_elbow_angle < 57  or \
            left_elbow_angle > 263 and left_elbow_angle < 267 and right_elbow_angle > 125 and right_elbow_angle < 129 or \
            left_elbow_angle > 220 and left_elbow_angle < 224 and right_elbow_angle > 99  and right_elbow_angle < 103 or \
            left_elbow_angle > 225 and left_elbow_angle < 229 and right_elbow_angle > 122 and right_elbow_angle < 126 or \
            left_elbow_angle > 0   and left_elbow_angle < 4   and right_elbow_angle > 356 and right_elbow_angle < 160 or \
            left_elbow_angle > 0   and left_elbow_angle < 3   and right_elbow_angle > 357 and right_elbow_angle < 360 or \
            left_elbow_angle > 288 and left_elbow_angle < 292 and right_elbow_angle > 66  and right_elbow_angle < 70  or \
            left_elbow_angle > 271 and left_elbow_angle < 275 and right_elbow_angle > 124 and right_elbow_angle < 128 or \
            left_elbow_angle > 206 and left_elbow_angle < 210 and right_elbow_angle > 169 and right_elbow_angle < 173 or \
            left_elbow_angle > 182 and left_elbow_angle < 186 and right_elbow_angle > 150 and right_elbow_angle < 154 or \
            left_elbow_angle > 278 and left_elbow_angle < 262 and right_elbow_angle > 159 and right_elbow_angle < 163 or \
            left_elbow_angle > 258 and left_elbow_angle < 262 and right_elbow_angle > 133 and right_elbow_angle < 137 or \
            left_elbow_angle > 231 and left_elbow_angle < 235 and right_elbow_angle > 114 and right_elbow_angle < 118 or \
            left_elbow_angle > 238 and left_elbow_angle < 242 and right_elbow_angle > 62  and right_elbow_angle < 66  or \
            left_elbow_angle > 223 and left_elbow_angle < 227 and right_elbow_angle > 148 and right_elbow_angle < 152 or \
            left_elbow_angle > 338 and left_elbow_angle < 342 and right_elbow_angle > 100 and right_elbow_angle < 104 or \
            left_elbow_angle > 108 and left_elbow_angle < 112 and right_elbow_angle > 76  and right_elbow_angle < 80  or \
            left_elbow_angle > 126 and left_elbow_angle < 130 and right_elbow_angle > 72  and right_elbow_angle < 76  or \
            left_elbow_angle > 8   and left_elbow_angle < 12  and right_elbow_angle > 66  and right_elbow_angle < 70  or \
            left_elbow_angle > 19  and left_elbow_angle < 23  and right_elbow_angle > 353 and right_elbow_angle < 357 or \
            left_elbow_angle > 18  and left_elbow_angle < 22  and right_elbow_angle > 29  and right_elbow_angle < 33  or \
            left_elbow_angle > 28  and left_elbow_angle < 32  and right_elbow_angle > 61  and right_elbow_angle < 65  or \
            left_elbow_angle > 43  and left_elbow_angle < 47  and right_elbow_angle > 104 and right_elbow_angle < 108 or \
            left_elbow_angle > 44  and left_elbow_angle < 48  and right_elbow_angle > 156 and right_elbow_angle < 160 or \
            left_elbow_angle > 131 and left_elbow_angle < 135 and right_elbow_angle > 167 and right_elbow_angle < 171 or \
            left_elbow_angle > 25  and left_elbow_angle < 29  and right_elbow_angle > 139 and right_elbow_angle < 143 or \
            left_elbow_angle > 49  and left_elbow_angle < 53  and right_elbow_angle > 121 and right_elbow_angle < 125 or \
            left_elbow_angle > 56  and left_elbow_angle < 60  and right_elbow_angle > 81  and right_elbow_angle < 85  or \
            left_elbow_angle > 39  and left_elbow_angle < 43  and right_elbow_angle > 117 and right_elbow_angle < 121 or \
            left_elbow_angle > 163 and left_elbow_angle < 167 and right_elbow_angle > 131 and right_elbow_angle < 135 or \
            left_elbow_angle > 167 and left_elbow_angle < 171 and right_elbow_angle > 100 and right_elbow_angle < 104 or \
            left_elbow_angle > 143 and left_elbow_angle < 147 and right_elbow_angle > 107 and right_elbow_angle < 111 or \
            left_elbow_angle > 110 and left_elbow_angle < 114 and right_elbow_angle > 139 and right_elbow_angle < 143 or \
            left_elbow_angle > 29  and left_elbow_angle < 33  and right_elbow_angle > 143 and right_elbow_angle < 147 or \
            left_elbow_angle > 18  and left_elbow_angle < 22  and right_elbow_angle > 141 and right_elbow_angle < 145 or \
            left_elbow_angle > 54  and left_elbow_angle < 58  and right_elbow_angle > 125 and right_elbow_angle < 129 or \
            left_elbow_angle > 0   and left_elbow_angle < 3   and right_elbow_angle > 66  and right_elbow_angle < 79  or \
            left_elbow_angle > 352 and left_elbow_angle < 358 and right_elbow_angle > 48  and right_elbow_angle < 52  or \
            left_elbow_angle > 347 and left_elbow_angle < 351 and right_elbow_angle > 77  and right_elbow_angle < 81  or \
            left_elbow_angle > 33  and left_elbow_angle < 37  and right_elbow_angle > 120 and right_elbow_angle < 124 or \
            left_elbow_angle > 290 and left_elbow_angle < 294 and right_elbow_angle > 155 and right_elbow_angle < 159 or \
            left_elbow_angle > 261 and left_elbow_angle < 265 and right_elbow_angle > 159 and right_elbow_angle < 163 or \
            left_elbow_angle > 275 and left_elbow_angle < 279 and right_elbow_angle > 157 and right_elbow_angle < 161 or \
            left_elbow_angle > 276 and left_elbow_angle < 280 and right_elbow_angle > 151 and right_elbow_angle < 155 or \
            left_elbow_angle > 268 and left_elbow_angle < 272 and right_elbow_angle > 146 and right_elbow_angle < 150 or \
            left_elbow_angle > 249 and left_elbow_angle < 253 and right_elbow_angle > 132 and right_elbow_angle < 136 or \
            left_elbow_angle > 140 and left_elbow_angle < 144 and right_elbow_angle > 40  and right_elbow_angle < 44  or \
            left_elbow_angle > 257 and left_elbow_angle < 261 and right_elbow_angle > 133 and right_elbow_angle < 137 or \
            left_elbow_angle > 255 and left_elbow_angle < 259 and right_elbow_angle > 129 and right_elbow_angle < 133 or \
            left_elbow_angle > 248 and left_elbow_angle < 252 and right_elbow_angle > 123 and right_elbow_angle < 127 or \
            left_elbow_angle > 249 and left_elbow_angle < 253 and right_elbow_angle > 124 and right_elbow_angle < 128 or \
            left_elbow_angle > 286 and left_elbow_angle < 290 and right_elbow_angle > 88  and right_elbow_angle < 92  or \
            left_elbow_angle > 333 and left_elbow_angle < 357 and right_elbow_angle > 69  and right_elbow_angle < 73  or \
            left_elbow_angle > 250 and left_elbow_angle < 254 and right_elbow_angle > 87  and right_elbow_angle < 91  or \
            left_elbow_angle > 322 and left_elbow_angle < 326 and right_elbow_angle > 99  and right_elbow_angle < 103 or \
            left_elbow_angle > 319 and left_elbow_angle < 322 and right_elbow_angle > 76  and right_elbow_angle < 80  or \
            left_elbow_angle > 309 and left_elbow_angle < 313 and right_elbow_angle > 71  and right_elbow_angle < 75  or \
            left_elbow_angle > 286 and left_elbow_angle < 290 and right_elbow_angle > 141 and right_elbow_angle < 145 or \
            left_elbow_angle > 133 and left_elbow_angle < 137 and right_elbow_angle > 127 and right_elbow_angle < 131 or \
            left_elbow_angle > 114 and left_elbow_angle < 118 and right_elbow_angle > 140 and right_elbow_angle < 144 or \
            left_elbow_angle > 62  and left_elbow_angle < 66  and right_elbow_angle > 149 and right_elbow_angle < 153 or \
            left_elbow_angle > 70  and left_elbow_angle < 74  and right_elbow_angle > 145 and right_elbow_angle < 149 or \
            left_elbow_angle > 44  and left_elbow_angle < 48  and right_elbow_angle > 91  and right_elbow_angle < 95  or \
            left_elbow_angle > 354 and left_elbow_angle < 358 and right_elbow_angle > 78  and right_elbow_angle < 82  or \
            left_elbow_angle > 98  and left_elbow_angle < 102 and right_elbow_angle > 134 and right_elbow_angle < 138 or \
            left_elbow_angle > 89  and left_elbow_angle < 93  and right_elbow_angle > 150 and right_elbow_angle < 154 or \
            left_elbow_angle > 36  and left_elbow_angle < 40  and right_elbow_angle > 138 and right_elbow_angle < 142 or \
            left_elbow_angle > 186 and left_elbow_angle < 190 and right_elbow_angle > 140 and right_elbow_angle < 144 or \
            left_elbow_angle > 266 and left_elbow_angle < 270 and right_elbow_angle > 140 and right_elbow_angle < 144 or \
            left_elbow_angle > 223 and left_elbow_angle < 227 and right_elbow_angle > 146 and right_elbow_angle < 150 or \
            left_elbow_angle > 210 and left_elbow_angle < 214 and right_elbow_angle > 142 and right_elbow_angle < 146 or \
            left_elbow_angle > 257 and left_elbow_angle < 261 and right_elbow_angle > 148 and right_elbow_angle < 152 or \
            left_elbow_angle > 96  and left_elbow_angle < 100 and right_elbow_angle > 238 and right_elbow_angle < 242 or \
            left_elbow_angle > 60  and left_elbow_angle < 64  and right_elbow_angle > 275 and right_elbow_angle < 279 or \
            left_elbow_angle > 296 and left_elbow_angle < 300 and right_elbow_angle > 282 and right_elbow_angle < 286 or \
            left_elbow_angle > 287 and left_elbow_angle < 291 and right_elbow_angle > 335 and right_elbow_angle < 339 or \
            left_elbow_angle > 294 and left_elbow_angle < 298 and right_elbow_angle > 340 and right_elbow_angle < 344 or \
            left_elbow_angle > 266 and left_elbow_angle < 270 and right_elbow_angle > 32  and right_elbow_angle < 36  or \
            left_elbow_angle > 271 and left_elbow_angle < 275 and right_elbow_angle > 48  and right_elbow_angle < 52  or \
            left_elbow_angle > 330 and left_elbow_angle < 334 and right_elbow_angle > 30  and right_elbow_angle < 34  or \
            left_elbow_angle > 327 and left_elbow_angle < 331 and right_elbow_angle > 43  and right_elbow_angle < 472 or \
            left_elbow_angle > 297 and left_elbow_angle < 301 and right_elbow_angle > 0   and right_elbow_angle < 2   or \
            left_elbow_angle > 298 and left_elbow_angle < 302 and right_elbow_angle > 340 and right_elbow_angle < 344 or \
            left_elbow_angle > 293 and left_elbow_angle < 297 and right_elbow_angle > 334 and right_elbow_angle < 338 or \
            left_elbow_angle > 282 and left_elbow_angle < 286 and right_elbow_angle > 323 and right_elbow_angle < 327 or \
            left_elbow_angle > 278 and left_elbow_angle < 282 and right_elbow_angle > 10  and right_elbow_angle < 14  or \
            left_elbow_angle > 107 and left_elbow_angle < 111 and right_elbow_angle > 269 and right_elbow_angle < 273 or \
            left_elbow_angle > 84  and left_elbow_angle < 88  and right_elbow_angle > 311 and right_elbow_angle < 315 or \
            left_elbow_angle > 346 and left_elbow_angle < 350 and right_elbow_angle > 77  and right_elbow_angle < 81  or \
            left_elbow_angle > 330 and left_elbow_angle < 334 and right_elbow_angle > 66  and right_elbow_angle < 70  or \
            left_elbow_angle > 319 and left_elbow_angle < 323 and right_elbow_angle > 110 and right_elbow_angle < 114 or \
            left_elbow_angle > 327 and left_elbow_angle < 331 and right_elbow_angle > 82  and right_elbow_angle < 86  or \
            left_elbow_angle > 330 and left_elbow_angle < 334 and right_elbow_angle > 43  and right_elbow_angle < 47  or \
            left_elbow_angle > 329 and left_elbow_angle < 333 and right_elbow_angle > 39  and right_elbow_angle < 43  or \
            left_elbow_angle > 331 and left_elbow_angle < 335 and right_elbow_angle > 29  and right_elbow_angle < 33  or \
            left_elbow_angle > 326 and left_elbow_angle < 330 and right_elbow_angle > 71  and right_elbow_angle < 75  or \
            left_elbow_angle > 325 and left_elbow_angle < 329 and right_elbow_angle > 87  and right_elbow_angle < 91  or \
            left_elbow_angle > 324 and left_elbow_angle < 328 and right_elbow_angle > 75  and right_elbow_angle < 79  or \
            left_elbow_angle > 317 and left_elbow_angle < 321 and right_elbow_angle > 118 and right_elbow_angle < 122 or \
            left_elbow_angle > 316 and left_elbow_angle < 320 and right_elbow_angle > 70  and right_elbow_angle < 74  or \
            left_elbow_angle > 96  and left_elbow_angle < 100 and right_elbow_angle > 255 and right_elbow_angle < 259 or \
            left_elbow_angle > 17  and left_elbow_angle < 21  and right_elbow_angle > 310 and right_elbow_angle < 314 or \
            left_elbow_angle > 243 and left_elbow_angle < 247 and right_elbow_angle > 69  and right_elbow_angle < 73  or \
            left_elbow_angle > 210 and left_elbow_angle < 214 and right_elbow_angle > 145 and right_elbow_angle < 149 or \
            left_elbow_angle > 262 and left_elbow_angle < 266 and right_elbow_angle > 62  and right_elbow_angle < 66  or \
            left_elbow_angle > 293 and left_elbow_angle < 297 and right_elbow_angle > 67  and right_elbow_angle < 71  or \
            left_elbow_angle > 337 and left_elbow_angle < 341 and right_elbow_angle > 31  and right_elbow_angle < 35  or \
            left_elbow_angle > 332 and left_elbow_angle < 337 and right_elbow_angle > 33  and right_elbow_angle < 37  or \
            left_elbow_angle > 305 and left_elbow_angle < 309 and right_elbow_angle > 85  and right_elbow_angle < 89  or \
            left_elbow_angle > 297 and left_elbow_angle < 301 and right_elbow_angle > 50  and right_elbow_angle < 54  or \
            left_elbow_angle > 315 and left_elbow_angle < 319 and right_elbow_angle > 10  and right_elbow_angle < 14  or \
            left_elbow_angle > 308 and left_elbow_angle < 312 and right_elbow_angle > 0   and right_elbow_angle < 4   or \
            left_elbow_angle > 257 and left_elbow_angle < 261 and right_elbow_angle > 135 and right_elbow_angle < 139 or \
            left_elbow_angle > 0 and left_elbow_angle < 30 and right_elbow_angle > 50 and right_elbow_angle < 70:
            if left_shoulder_angle > 50 and left_shoulder_angle < 70 and right_shoulder_angle > 50 and right_shoulder_angle < 70 or \
                left_shoulder_angle > 30 and left_shoulder_angle < 50 and right_shoulder_angle > 20 and right_shoulder_angle < 40 or \
                left_shoulder_angle > 50 and left_shoulder_angle < 70 and right_shoulder_angle > 40 and right_shoulder_angle < 70 or \
                left_shoulder_angle > 20 and left_shoulder_angle < 40 and right_shoulder_angle > 0 and right_shoulder_angle < 30 or\
                left_shoulder_angle > 0 and left_shoulder_angle < 30 and right_shoulder_angle > 0 and right_shoulder_angle < 30 or\
                left_shoulder_angle > 0 and left_shoulder_angle < 30 and right_shoulder_angle > 330 and right_shoulder_angle < 360 or\
                left_shoulder_angle > 30 and left_shoulder_angle < 60 and right_shoulder_angle > 30 and right_shoulder_angle < 60 or\
                left_shoulder_angle >  58 and left_shoulder_angle < 62  and right_shoulder_angle > 75  and right_shoulder_angle < 79  or \
                left_shoulder_angle > 59  and left_shoulder_angle < 63  and right_shoulder_angle > 80  and right_shoulder_angle < 84  or \
                left_shoulder_angle > 61  and left_shoulder_angle < 65  and right_shoulder_angle > 80  and right_shoulder_angle < 84  or \
                left_shoulder_angle > 49  and left_shoulder_angle < 53  and right_shoulder_angle > 30  and right_shoulder_angle < 34  or \
                left_shoulder_angle > 40  and left_shoulder_angle < 44  and right_shoulder_angle > 337 and right_shoulder_angle < 341 or \
                left_shoulder_angle > 27  and left_shoulder_angle < 31  and right_shoulder_angle > 243 and right_shoulder_angle < 347 or \
                left_shoulder_angle > 36  and left_shoulder_angle < 40  and right_shoulder_angle > 331 and right_shoulder_angle < 335 or \
                left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 343 and right_shoulder_angle < 347 or \
                left_shoulder_angle > 24  and left_shoulder_angle < 28  and right_shoulder_angle > 3   and right_shoulder_angle < 7   or \
                left_shoulder_angle > 20  and left_shoulder_angle < 24  and right_shoulder_angle > 14  and right_shoulder_angle < 18  or \
                left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 0   and right_shoulder_angle < 3   or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 19  and left_shoulder_angle < 23  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 12  and left_shoulder_angle < 16  and right_shoulder_angle > 346 and right_shoulder_angle < 350 or \
                left_shoulder_angle > 14  and left_shoulder_angle < 18  and right_shoulder_angle > 344 and right_shoulder_angle < 348 or \
                left_shoulder_angle > 15  and left_shoulder_angle < 19  and right_shoulder_angle > 349 and right_shoulder_angle < 353 or \
                left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 346 and right_shoulder_angle < 350 or \
                left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 19  and left_shoulder_angle < 23  and right_shoulder_angle > 343 and right_shoulder_angle < 347 or \
                left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 25  and right_shoulder_angle < 29  or \
                left_shoulder_angle > 357 and left_shoulder_angle < 360 and right_shoulder_angle > 353 and right_shoulder_angle < 357 or \
                left_shoulder_angle > 11  and left_shoulder_angle < 15  and right_shoulder_angle > 6   and right_shoulder_angle < 10  or \
                left_shoulder_angle > 11  and left_shoulder_angle < 15  and right_shoulder_angle > 25  and right_shoulder_angle < 29  or \
                left_shoulder_angle > 12  and left_shoulder_angle < 16  and right_shoulder_angle > 6   and right_shoulder_angle < 10  or \
                left_shoulder_angle > 30  and left_shoulder_angle < 34  and right_shoulder_angle > 304 and right_shoulder_angle < 308 or \
                left_shoulder_angle > 0   and left_shoulder_angle < 2   and right_shoulder_angle > 333 and right_shoulder_angle < 357 or \
                left_shoulder_angle > 12  and left_shoulder_angle < 16  and right_shoulder_angle > 11  and right_shoulder_angle < 15  or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 22  and right_shoulder_angle < 26  or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 23  and right_shoulder_angle < 27  or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 23  and right_shoulder_angle < 27  or \
                left_shoulder_angle > 17  and left_shoulder_angle < 21  and right_shoulder_angle > 10  and right_shoulder_angle < 14  or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 357 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 0   and left_shoulder_angle < 2   and right_shoulder_angle > 356 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 41  and left_shoulder_angle < 45  and right_shoulder_angle > 0   and right_shoulder_angle < 4   or \
                left_shoulder_angle > 34  and left_shoulder_angle < 38  and right_shoulder_angle > 335 and right_shoulder_angle < 339 or \
                left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 332 and right_shoulder_angle < 336 or \
                left_shoulder_angle > 27  and left_shoulder_angle < 31  and right_shoulder_angle > 329 and right_shoulder_angle < 333 or \
                left_shoulder_angle > 31  and left_shoulder_angle < 35  and right_shoulder_angle > 249 and right_shoulder_angle < 353 or \
                left_shoulder_angle > 21  and left_shoulder_angle < 25  and right_shoulder_angle > 16  and right_shoulder_angle < 20  or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 17  and right_shoulder_angle < 21  or \
                left_shoulder_angle > 30  and left_shoulder_angle < 34  and right_shoulder_angle > 352 and right_shoulder_angle < 356 or \
                left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 346 and right_shoulder_angle < 350 or \
                left_shoulder_angle > 28  and left_shoulder_angle < 32  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 27  and left_shoulder_angle < 31  and right_shoulder_angle > 348 and right_shoulder_angle < 352 or \
                left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 350 and right_shoulder_angle < 354 or \
                left_shoulder_angle > 27  and left_shoulder_angle < 31  and right_shoulder_angle > 349 and right_shoulder_angle < 353 or \
                left_shoulder_angle > 25  and left_shoulder_angle < 29  and right_shoulder_angle > 349 and right_shoulder_angle < 353 or \
                left_shoulder_angle > 38  and left_shoulder_angle < 42  and right_shoulder_angle > 44  and right_shoulder_angle < 48  or \
                left_shoulder_angle > 0   and left_shoulder_angle < 3   and right_shoulder_angle > 25  and right_shoulder_angle < 29  or \
                left_shoulder_angle > 350 and left_shoulder_angle < 354 and right_shoulder_angle > 11  and right_shoulder_angle < 15  or \
                left_shoulder_angle > 332 and left_shoulder_angle < 336 and right_shoulder_angle > 4   and right_shoulder_angle < 8   or \
                left_shoulder_angle > 340 and left_shoulder_angle < 344 and right_shoulder_angle > 1   and right_shoulder_angle < 5   or \
                left_shoulder_angle > 338 and left_shoulder_angle < 342 and right_shoulder_angle > 25  and right_shoulder_angle < 29  or \
                left_shoulder_angle > 332 and left_shoulder_angle < 336 and right_shoulder_angle > 24  and right_shoulder_angle < 28  or \
                left_shoulder_angle > 351 and left_shoulder_angle < 355 and right_shoulder_angle > 23  and right_shoulder_angle < 27  or \
                left_shoulder_angle > 16  and left_shoulder_angle < 20  and right_shoulder_angle > 22  and right_shoulder_angle < 26  or \
                left_shoulder_angle > 14  and left_shoulder_angle < 18  and right_shoulder_angle > 21  and right_shoulder_angle < 25  or \
                left_shoulder_angle > 357 and left_shoulder_angle < 360 and right_shoulder_angle > 14  and right_shoulder_angle < 18  or \
                left_shoulder_angle > 353 and left_shoulder_angle < 357 and right_shoulder_angle > 24  and right_shoulder_angle < 28  or \
                left_shoulder_angle > 357 and left_shoulder_angle < 360 and right_shoulder_angle > 14  and right_shoulder_angle < 18  or \
                left_shoulder_angle > 249 and left_shoulder_angle < 253 and right_shoulder_angle > 20  and right_shoulder_angle < 24  or \
                left_shoulder_angle > 248 and left_shoulder_angle < 252 and right_shoulder_angle > 18  and right_shoulder_angle < 22  or \
                left_shoulder_angle > 345 and left_shoulder_angle < 349 and right_shoulder_angle > 11  and right_shoulder_angle < 15  or \
                left_shoulder_angle > 349 and left_shoulder_angle < 353 and right_shoulder_angle > 8   and right_shoulder_angle < 12  or \
                left_shoulder_angle > 348 and left_shoulder_angle < 352 and right_shoulder_angle > 9   and right_shoulder_angle < 13  or \
                left_shoulder_angle > 353 and left_shoulder_angle < 357 and right_shoulder_angle > 13  and right_shoulder_angle < 17  or \
                left_shoulder_angle > 0   and left_shoulder_angle < 3   and right_shoulder_angle > 9   and right_shoulder_angle < 15  or \
                left_shoulder_angle > 11  and left_shoulder_angle < 15  and right_shoulder_angle > 10  and right_shoulder_angle < 14  or \
                left_shoulder_angle > 37  and left_shoulder_angle < 41  and right_shoulder_angle > 20  and right_shoulder_angle < 24  or \
                left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 1   and right_shoulder_angle < 5   or \
                left_shoulder_angle > 357 and left_shoulder_angle < 360 and right_shoulder_angle > 351 and right_shoulder_angle < 355 or \
                left_shoulder_angle > 1   and left_shoulder_angle < 5   and right_shoulder_angle > 347 and right_shoulder_angle < 351 or \
                left_shoulder_angle > 7   and left_shoulder_angle < 11  and right_shoulder_angle > 6   and right_shoulder_angle < 10  or \
                left_shoulder_angle > 4   and left_shoulder_angle < 8   and right_shoulder_angle > 0   and right_shoulder_angle < 2   or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 1   and right_shoulder_angle < 5   or \
                left_shoulder_angle > 7   and left_shoulder_angle < 11  and right_shoulder_angle > 1   and right_shoulder_angle < 5   or \
                left_shoulder_angle > 0   and left_shoulder_angle < 4   and right_shoulder_angle > 0   and right_shoulder_angle < 3   or \
                left_shoulder_angle > 11  and left_shoulder_angle < 15  and right_shoulder_angle > 19  and right_shoulder_angle < 23  or \
                left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 46  and right_shoulder_angle < 50  or \
                left_shoulder_angle > 7   and left_shoulder_angle < 11  and right_shoulder_angle > 26  and right_shoulder_angle < 30  or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 0   and right_shoulder_angle < 3   or \
                left_shoulder_angle > 1   and left_shoulder_angle < 5   and right_shoulder_angle > 0   and right_shoulder_angle < 2   or \
                left_shoulder_angle > 357 and left_shoulder_angle < 360 and right_shoulder_angle > 4   and right_shoulder_angle < 8   or \
                left_shoulder_angle > 347 and left_shoulder_angle < 351 and right_shoulder_angle > 6   and right_shoulder_angle < 10  or \
                left_shoulder_angle > 340 and left_shoulder_angle < 344 and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 347 and left_shoulder_angle < 351 and right_shoulder_angle > 6   and right_shoulder_angle < 10  or \
                left_shoulder_angle > 1   and left_shoulder_angle < 5   and right_shoulder_angle > 15  and right_shoulder_angle < 19  or \
                left_shoulder_angle > 0   and left_shoulder_angle < 4   and right_shoulder_angle > 9   and right_shoulder_angle < 13  or \
                left_shoulder_angle > 0   and left_shoulder_angle < 0   and right_shoulder_angle > 8   and right_shoulder_angle < 12  or \
                left_shoulder_angle > 357 and left_shoulder_angle < 360 and right_shoulder_angle > 11  and right_shoulder_angle < 15  or \
                left_shoulder_angle > 354 and left_shoulder_angle < 358 and right_shoulder_angle > 14  and right_shoulder_angle < 18  or \
                left_shoulder_angle > 356 and left_shoulder_angle < 360 and right_shoulder_angle > 15  and right_shoulder_angle < 19  or \
                left_shoulder_angle > 354 and left_shoulder_angle < 358 and right_shoulder_angle > 17  and right_shoulder_angle < 21  or \
                left_shoulder_angle > 349 and left_shoulder_angle < 353 and right_shoulder_angle > 19  and right_shoulder_angle < 23  or \
                left_shoulder_angle > 351 and left_shoulder_angle < 355 and right_shoulder_angle > 14  and right_shoulder_angle < 18  or \
                left_shoulder_angle > 354 and left_shoulder_angle < 358 and right_shoulder_angle > 12  and right_shoulder_angle < 16  or \
                left_shoulder_angle > 354 and left_shoulder_angle < 358 and right_shoulder_angle > 12  and right_shoulder_angle < 16  or \
                left_shoulder_angle > 0   and left_shoulder_angle < 2   and right_shoulder_angle > 13  and right_shoulder_angle < 17  or \
                left_shoulder_angle > 2   and left_shoulder_angle < 6   and right_shoulder_angle > 14  and right_shoulder_angle < 18  or \
                left_shoulder_angle > 355 and left_shoulder_angle < 359 and right_shoulder_angle > 9   and right_shoulder_angle < 13  or \
                left_shoulder_angle > 354 and left_shoulder_angle < 358 and right_shoulder_angle > 15  and right_shoulder_angle < 19  or \
                left_shoulder_angle > 52  and left_shoulder_angle < 56  and right_shoulder_angle > 55  and right_shoulder_angle < 59  or \
                left_shoulder_angle > 84  and left_shoulder_angle < 88  and right_shoulder_angle > 86  and right_shoulder_angle < 90  or \
                left_shoulder_angle > 78  and left_shoulder_angle < 82  and right_shoulder_angle > 64  and right_shoulder_angle < 68  or \
                left_shoulder_angle > 69  and left_shoulder_angle < 73  and right_shoulder_angle > 37  and right_shoulder_angle < 41  or \
                left_shoulder_angle > 59  and left_shoulder_angle < 63  and right_shoulder_angle > 353 and right_shoulder_angle < 357 or \
                left_shoulder_angle > 19  and left_shoulder_angle < 23  and right_shoulder_angle > 338 and right_shoulder_angle < 342 or \
                left_shoulder_angle > 8   and left_shoulder_angle < 12  and right_shoulder_angle > 343 and right_shoulder_angle < 347 or \
                left_shoulder_angle > 35  and left_shoulder_angle < 39  and right_shoulder_angle > 314 and right_shoulder_angle < 318 or \
                left_shoulder_angle > 40  and left_shoulder_angle < 44  and right_shoulder_angle > 329 and right_shoulder_angle < 333 or \
                left_shoulder_angle > 17  and left_shoulder_angle < 21  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 50  and left_shoulder_angle < 54  and right_shoulder_angle > 62  and right_shoulder_angle < 66  or \
                left_shoulder_angle > 62  and left_shoulder_angle < 66  and right_shoulder_angle > 11  and right_shoulder_angle < 15  or \
                left_shoulder_angle > 44  and left_shoulder_angle < 48  and right_shoulder_angle > 331 and right_shoulder_angle < 335 or \
                left_shoulder_angle > 38  and left_shoulder_angle < 42  and right_shoulder_angle > 333 and right_shoulder_angle < 337 or \
                left_shoulder_angle > 32  and left_shoulder_angle < 36  and right_shoulder_angle > 235 and right_shoulder_angle < 239 or \
                left_shoulder_angle > 38  and left_shoulder_angle < 42  and right_shoulder_angle > 240 and right_shoulder_angle < 244 or \
                left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 230 and right_shoulder_angle < 234 or \
                left_shoulder_angle > 45  and left_shoulder_angle < 49  and right_shoulder_angle > 322 and right_shoulder_angle < 326 or \
                left_shoulder_angle > 31  and left_shoulder_angle < 35  and right_shoulder_angle > 341 and right_shoulder_angle < 345 or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 343 and right_shoulder_angle < 347 or \
                left_shoulder_angle > 67  and left_shoulder_angle < 71  and right_shoulder_angle > 70  and right_shoulder_angle < 74  or \
                left_shoulder_angle > 351 and left_shoulder_angle < 355 and right_shoulder_angle > 16  and right_shoulder_angle < 20  or \
                left_shoulder_angle > 335 and left_shoulder_angle < 339 and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 333 and left_shoulder_angle < 337 and right_shoulder_angle > 354 and right_shoulder_angle < 358 or \
                left_shoulder_angle > 307 and left_shoulder_angle < 311 and right_shoulder_angle > 342 and right_shoulder_angle < 346 or \
                left_shoulder_angle > 330 and left_shoulder_angle < 334 and right_shoulder_angle > 348 and right_shoulder_angle < 352 or \
                left_shoulder_angle > 48  and left_shoulder_angle < 52  and right_shoulder_angle > 54  and right_shoulder_angle < 58  or \
                left_shoulder_angle > 52  and left_shoulder_angle < 56  and right_shoulder_angle > 59  and right_shoulder_angle < 63  or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 12  and right_shoulder_angle < 16  or \
                left_shoulder_angle > 351 and left_shoulder_angle < 355 and right_shoulder_angle > 0   and right_shoulder_angle < 4   or \
                left_shoulder_angle > 0   and left_shoulder_angle < 2   and right_shoulder_angle > 353 and right_shoulder_angle < 357 or \
                left_shoulder_angle > 348 and left_shoulder_angle < 352 and right_shoulder_angle > 0   and right_shoulder_angle < 2   or \
                left_shoulder_angle > 349 and left_shoulder_angle < 353 and right_shoulder_angle > 356 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 349 and left_shoulder_angle < 353 and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 305 and left_shoulder_angle < 309 and right_shoulder_angle > 346 and right_shoulder_angle < 350 or \
                left_shoulder_angle > 337 and left_shoulder_angle < 341 and right_shoulder_angle > 2   and right_shoulder_angle < 6   or \
                left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 333 and right_shoulder_angle < 337 or \
                left_shoulder_angle > 89  and left_shoulder_angle < 93  and right_shoulder_angle > 343 and right_shoulder_angle < 347 or \
                left_shoulder_angle > 104 and left_shoulder_angle < 108 and right_shoulder_angle > 342 and right_shoulder_angle < 346 or \
                left_shoulder_angle > 108 and left_shoulder_angle < 112 and right_shoulder_angle > 344 and right_shoulder_angle < 348 or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 326 and right_shoulder_angle < 330 or \
                left_shoulder_angle > 60  and left_shoulder_angle < 64  and right_shoulder_angle > 43  and right_shoulder_angle < 47  or \
                left_shoulder_angle > 45  and left_shoulder_angle < 49  and right_shoulder_angle > 3   and right_shoulder_angle < 7   or \
                left_shoulder_angle > 32  and left_shoulder_angle < 36  and right_shoulder_angle > 350 and right_shoulder_angle < 354 or \
                left_shoulder_angle > 38  and left_shoulder_angle < 42  and right_shoulder_angle > 336 and right_shoulder_angle < 340 or \
                left_shoulder_angle > 31  and left_shoulder_angle < 35  and right_shoulder_angle > 344 and right_shoulder_angle < 348 or \
                left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 346 and right_shoulder_angle < 350 or \
                left_shoulder_angle > 15  and left_shoulder_angle < 19  and right_shoulder_angle > 341 and right_shoulder_angle < 345 or \
                left_shoulder_angle > 22  and left_shoulder_angle < 26  and right_shoulder_angle > 336 and right_shoulder_angle < 340 or \
                left_shoulder_angle > 23  and left_shoulder_angle < 27  and right_shoulder_angle > 338 and right_shoulder_angle < 342 or \
                left_shoulder_angle > 62  and left_shoulder_angle < 66  and right_shoulder_angle > 9   and right_shoulder_angle < 13  or \
                left_shoulder_angle > 41  and left_shoulder_angle < 45  and right_shoulder_angle > 336 and right_shoulder_angle < 340 or \
                left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 313 and right_shoulder_angle < 317 or \
                left_shoulder_angle > 49  and left_shoulder_angle < 53  and right_shoulder_angle > 315 and right_shoulder_angle < 319 or \
                left_shoulder_angle > 59  and left_shoulder_angle < 63  and right_shoulder_angle > 316 and right_shoulder_angle < 321 or \
                left_shoulder_angle > 47  and left_shoulder_angle < 51  and right_shoulder_angle > 325 and right_shoulder_angle < 329 or \
                left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 354 and right_shoulder_angle < 358 or \
                left_shoulder_angle > 25  and left_shoulder_angle < 29  and right_shoulder_angle > 1   and right_shoulder_angle < 5   or \
                left_shoulder_angle > 43  and left_shoulder_angle < 47  and right_shoulder_angle > 355 and right_shoulder_angle < 359 or \
                left_shoulder_angle > 47  and left_shoulder_angle < 51  and right_shoulder_angle > 326 and right_shoulder_angle < 330 or \
                left_shoulder_angle > 39  and left_shoulder_angle < 43  and right_shoulder_angle > 314 and right_shoulder_angle < 318 or \
                left_shoulder_angle > 32  and left_shoulder_angle < 36  and right_shoulder_angle > 322 and right_shoulder_angle < 326 or \
                left_shoulder_angle > 30  and left_shoulder_angle < 34  and right_shoulder_angle > 323 and right_shoulder_angle < 327 or \
                left_shoulder_angle > 18  and left_shoulder_angle < 22  and right_shoulder_angle > 323 and right_shoulder_angle < 327 or \
                left_shoulder_angle > 8   and left_shoulder_angle < 12  and right_shoulder_angle > 329 and right_shoulder_angle < 333 or \
                left_shoulder_angle > 26  and left_shoulder_angle < 30  and right_shoulder_angle > 340 and right_shoulder_angle < 344 or \
                left_shoulder_angle > 44  and left_shoulder_angle < 48  and right_shoulder_angle > 3   and right_shoulder_angle < 7   or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 219 and right_shoulder_angle < 323 or \
                left_shoulder_angle > 332 and left_shoulder_angle < 336 and right_shoulder_angle > 329 and right_shoulder_angle < 333 or \
                left_shoulder_angle > 329 and left_shoulder_angle < 333 and right_shoulder_angle > 335 and right_shoulder_angle < 339 or \
                left_shoulder_angle > 328 and left_shoulder_angle < 332 and right_shoulder_angle > 333 and right_shoulder_angle < 337 or \
                left_shoulder_angle > 0   and left_shoulder_angle < 4   and right_shoulder_angle > 12  and right_shoulder_angle < 16  or \
                left_shoulder_angle > 14  and left_shoulder_angle < 18  and right_shoulder_angle > 357 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 12  and left_shoulder_angle < 16  and right_shoulder_angle > 1   and right_shoulder_angle < 5   or \
                left_shoulder_angle > 7   and left_shoulder_angle < 11  and right_shoulder_angle > 349 and right_shoulder_angle < 353 or \
                left_shoulder_angle > 9   and left_shoulder_angle < 13  and right_shoulder_angle > 12  and right_shoulder_angle < 16  or \
                left_shoulder_angle > 9   and left_shoulder_angle < 13  and right_shoulder_angle > 11  and right_shoulder_angle < 15  or \
                left_shoulder_angle > 2   and left_shoulder_angle < 6   and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 40  and left_shoulder_angle < 44  and right_shoulder_angle > 328 and right_shoulder_angle < 332 or \
                left_shoulder_angle > 48  and left_shoulder_angle < 52  and right_shoulder_angle > 314 and right_shoulder_angle < 318 or \
                left_shoulder_angle > 44  and left_shoulder_angle < 48  and right_shoulder_angle > 309 and right_shoulder_angle < 313 or \
                left_shoulder_angle > 39  and left_shoulder_angle < 43  and right_shoulder_angle > 307 and right_shoulder_angle < 311 or \
                left_shoulder_angle > 40  and left_shoulder_angle < 44  and right_shoulder_angle > 310 and right_shoulder_angle < 314 or \
                left_shoulder_angle > 43  and left_shoulder_angle < 47  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 35  and left_shoulder_angle < 39  and right_shoulder_angle > 0   and right_shoulder_angle < 2   or \
                left_shoulder_angle > 52  and left_shoulder_angle < 56  and right_shoulder_angle > 320 and right_shoulder_angle < 324 or \
                left_shoulder_angle > 48  and left_shoulder_angle < 52  and right_shoulder_angle > 312 and right_shoulder_angle < 316 or \
                left_shoulder_angle > 40  and left_shoulder_angle < 44  and right_shoulder_angle > 324 and right_shoulder_angle < 328 or \
                left_shoulder_angle > 54  and left_shoulder_angle < 58  and right_shoulder_angle > 325 and right_shoulder_angle < 329 or \
                left_shoulder_angle > 55  and left_shoulder_angle < 59  and right_shoulder_angle > 323 and right_shoulder_angle < 327 or \
                left_shoulder_angle > 60  and left_shoulder_angle < 64  and right_shoulder_angle > 318 and right_shoulder_angle < 322 or \
                left_shoulder_angle > 61  and left_shoulder_angle < 65  and right_shoulder_angle > 319 and right_shoulder_angle < 323 or \
                left_shoulder_angle > 67  and left_shoulder_angle < 71  and right_shoulder_angle > 314 and right_shoulder_angle < 318 or \
                left_shoulder_angle > 37  and left_shoulder_angle < 41  and right_shoulder_angle > 330 and right_shoulder_angle < 334 or \
                left_shoulder_angle > 54  and left_shoulder_angle < 58  and right_shoulder_angle > 62  and right_shoulder_angle < 66  or \
                left_shoulder_angle > 28  and left_shoulder_angle < 32  and right_shoulder_angle > 41  and right_shoulder_angle < 45  or \
                left_shoulder_angle > 328 and left_shoulder_angle < 332 and right_shoulder_angle > 35  and right_shoulder_angle < 39  or \
                left_shoulder_angle > 326 and left_shoulder_angle < 330 and right_shoulder_angle > 16  and right_shoulder_angle < 20  or \
                left_shoulder_angle > 334 and left_shoulder_angle < 338 and right_shoulder_angle > 356 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 325 and left_shoulder_angle < 329 and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 335 and left_shoulder_angle < 339 and right_shoulder_angle > 354 and right_shoulder_angle < 358 or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 3   and right_shoulder_angle < 7   or \
                left_shoulder_angle > 322 and left_shoulder_angle < 326 and right_shoulder_angle > 42  and right_shoulder_angle < 46  or \
                left_shoulder_angle > 357 and left_shoulder_angle < 361 and right_shoulder_angle > 6   and right_shoulder_angle < 10  or \
                left_shoulder_angle > 356 and left_shoulder_angle < 360 and right_shoulder_angle > 9   and right_shoulder_angle < 13  or \
                left_shoulder_angle > 0   and left_shoulder_angle < 3   and right_shoulder_angle > 10  and right_shoulder_angle < 14  or \
                left_shoulder_angle > 350 and left_shoulder_angle < 354 and right_shoulder_angle > 16  and right_shoulder_angle < 20  or \
                left_shoulder_angle > 347 and left_shoulder_angle < 351 and right_shoulder_angle > 9   and right_shoulder_angle < 13  or \
                left_shoulder_angle > 64  and left_shoulder_angle < 68  and right_shoulder_angle > 31  and right_shoulder_angle < 35  or \
                left_shoulder_angle > 52  and left_shoulder_angle < 56  and right_shoulder_angle > 42  and right_shoulder_angle < 46  or \
                left_shoulder_angle > 9   and left_shoulder_angle < 13  and right_shoulder_angle > 337 and right_shoulder_angle < 341 or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 357 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 3   and left_shoulder_angle < 7   and right_shoulder_angle > 328 and right_shoulder_angle < 332 or \
                left_shoulder_angle > 10  and left_shoulder_angle < 14  and right_shoulder_angle > 341 and right_shoulder_angle < 345 or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 357 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 3   and left_shoulder_angle < 7   and right_shoulder_angle > 3   and right_shoulder_angle < 7   or \
                left_shoulder_angle > 8   and left_shoulder_angle < 12  and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 9   and left_shoulder_angle < 13  and right_shoulder_angle > 357 and right_shoulder_angle < 360 or \
                left_shoulder_angle > 4   and left_shoulder_angle < 8   and right_shoulder_angle > 1   and right_shoulder_angle < 5   or \
                left_shoulder_angle > 9   and left_shoulder_angle < 13  and right_shoulder_angle > 352 and right_shoulder_angle < 356 or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 345 and right_shoulder_angle < 349 or \
                left_shoulder_angle > 7   and left_shoulder_angle < 11  and right_shoulder_angle > 347 and right_shoulder_angle < 351 or \
                left_shoulder_angle > 45  and left_shoulder_angle < 49  and right_shoulder_angle > 51  and right_shoulder_angle < 55  or \
                left_shoulder_angle > 10  and left_shoulder_angle < 14  and right_shoulder_angle > 29  and right_shoulder_angle < 33  or \
                left_shoulder_angle > 302 and left_shoulder_angle < 306 and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 284 and left_shoulder_angle < 288 and right_shoulder_angle > 289 and right_shoulder_angle < 292 or \
                left_shoulder_angle > 325 and left_shoulder_angle < 329 and right_shoulder_angle > 340 and right_shoulder_angle < 344 or \
                left_shoulder_angle > 339 and left_shoulder_angle < 343 and right_shoulder_angle > 349 and right_shoulder_angle < 353 or \
                left_shoulder_angle > 0   and left_shoulder_angle < 4   and right_shoulder_angle > 0   and right_shoulder_angle < 4   or \
                left_shoulder_angle > 6   and left_shoulder_angle < 10  and right_shoulder_angle > 6   and right_shoulder_angle < 10  or \
                left_shoulder_angle > 2   and left_shoulder_angle < 6   and right_shoulder_angle > 10  and right_shoulder_angle < 14  or \
                left_shoulder_angle > 4   and left_shoulder_angle < 8   and right_shoulder_angle > 3   and right_shoulder_angle < 7   or \
                left_shoulder_angle > 3   and left_shoulder_angle < 7   and right_shoulder_angle > 4   and right_shoulder_angle < 8   or \
                left_shoulder_angle > 0   and left_shoulder_angle < 4   and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 5 and left_shoulder_angle < 30 and right_shoulder_angle > 20 and right_shoulder_angle < 40:
                if left_knee_angle > 170 and left_knee_angle < 200 and right_knee_angle > 170 and right_knee_angle < 200 or \
                    left_knee_angle > 70 and left_knee_angle < 100 and right_knee_angle > 160 and right_knee_angle < 200 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 187 and right_knee_angle < 191 or \
                    left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 189 and right_knee_angle < 193 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 180 and right_knee_angle < 194 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 188 and right_knee_angle < 192 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 182 and right_knee_angle < 186 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 184 and right_knee_angle < 188 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 187 and right_knee_angle < 191 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 190 and right_knee_angle < 194 or \
                    left_knee_angle > 189 and left_knee_angle < 193 and right_knee_angle > 199 and right_knee_angle < 203 or \
                    left_knee_angle > 185 and left_knee_angle < 189 and right_knee_angle > 196 and right_knee_angle < 200 or \
                    left_knee_angle > 187 and left_knee_angle < 191 and right_knee_angle > 195 and right_knee_angle < 199 or \
                    left_knee_angle > 185 and left_knee_angle < 189 and right_knee_angle > 196 and right_knee_angle < 200 or \
                    left_knee_angle > 187 and left_knee_angle < 191 and right_knee_angle > 194 and right_knee_angle < 198 or \
                    left_knee_angle > 187 and left_knee_angle < 191 and right_knee_angle > 197 and right_knee_angle < 201 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 178 and right_knee_angle < 182 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 178 and right_knee_angle < 182 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 180 and right_knee_angle < 184 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 182 and right_knee_angle < 186 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 180 and right_knee_angle < 184 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 123 and left_knee_angle < 127 and right_knee_angle > 187 and right_knee_angle < 191 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 192 and right_knee_angle < 196 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 189 and right_knee_angle < 193 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 186 and left_knee_angle < 190 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 183 and left_knee_angle < 187 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 185 and left_knee_angle < 189 and right_knee_angle > 187 and right_knee_angle < 191 or \
                    left_knee_angle > 189 and left_knee_angle < 193 and right_knee_angle > 199 and right_knee_angle < 203 or \
                    left_knee_angle > 192 and left_knee_angle < 196 and right_knee_angle > 208 and right_knee_angle < 212 or \
                    left_knee_angle > 196 and left_knee_angle < 200 and right_knee_angle > 214 and right_knee_angle < 218 or \
                    left_knee_angle > 199 and left_knee_angle < 203 and right_knee_angle > 216 and right_knee_angle < 220 or \
                    left_knee_angle > 198 and left_knee_angle < 202 and right_knee_angle > 213 and right_knee_angle < 217 or \
                    left_knee_angle > 186 and left_knee_angle < 190 and right_knee_angle > 197 and right_knee_angle < 201 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 178 and right_knee_angle < 182 or \
                    left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 169 and right_knee_angle < 173 or \
                    left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 168 and right_knee_angle < 172 or \
                    left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 169 and right_knee_angle < 173 or \
                    left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 167 and left_knee_angle < 171 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 167 and left_knee_angle < 171 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 166 and right_knee_angle < 170 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 163 and right_knee_angle < 167 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 166 and right_knee_angle < 170 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 168 and right_knee_angle < 172 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 169 and right_knee_angle < 173 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 170 and left_knee_angle < 174 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 169 and right_knee_angle < 173 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 167 and right_knee_angle < 171 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 170 and left_knee_angle < 174 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 167 and right_knee_angle < 171 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 168 and right_knee_angle < 172 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 166 and right_knee_angle < 170 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 165 and right_knee_angle < 169 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 160 and right_knee_angle < 164 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 162 and right_knee_angle < 166 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 165 and right_knee_angle < 169 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 165 and right_knee_angle < 169 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 167 and right_knee_angle < 171 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 167 and right_knee_angle < 171 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 168 and right_knee_angle < 172 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 178 and right_knee_angle < 182 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 180 and right_knee_angle < 184 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 330 and left_knee_angle < 334 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 183 and left_knee_angle < 187 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 178 and right_knee_angle < 182 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 187 and left_knee_angle < 191 and right_knee_angle > 178 and right_knee_angle < 182 or \
                    left_knee_angle > 185 and left_knee_angle < 189 and right_knee_angle > 184 and right_knee_angle < 188 or \
                    left_knee_angle > 186 and left_knee_angle < 190 and right_knee_angle > 190 and right_knee_angle < 194 or \
                    left_knee_angle > 187 and left_knee_angle < 191 and right_knee_angle > 192 and right_knee_angle < 196 or \
                    left_knee_angle > 186 and left_knee_angle < 190 and right_knee_angle > 192 and right_knee_angle < 196 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 184 and right_knee_angle < 188 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 185 and left_knee_angle < 189 and right_knee_angle > 189 and right_knee_angle < 193 or \
                    left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 168 and right_knee_angle < 172 or \
                    left_knee_angle > 183 and left_knee_angle < 187 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 182 and left_knee_angle < 185 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 180 and left_knee_angle < 184 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 180 and right_knee_angle < 184 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 166 and right_knee_angle < 170 or \
                    left_knee_angle > 163 and left_knee_angle < 167 and right_knee_angle > 164 and right_knee_angle < 168 or \
                    left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 165 and right_knee_angle < 169 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 162 and right_knee_angle < 166 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 167 and right_knee_angle < 171 or \
                    left_knee_angle > 172 and left_knee_angle < 176 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 169 and right_knee_angle < 173 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 180 and right_knee_angle < 184 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 182 and right_knee_angle < 186 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 183 and left_knee_angle < 187 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 184 and right_knee_angle < 188 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 191 and right_knee_angle < 195 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 182 and right_knee_angle < 186 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 182 and right_knee_angle < 186 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 184 and left_knee_angle < 188 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 188 and left_knee_angle < 192 and right_knee_angle > 191 and right_knee_angle < 195 or \
                    left_knee_angle > 186 and left_knee_angle < 200 and right_knee_angle > 190 and right_knee_angle < 194 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 190 and right_knee_angle < 194 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 181 and left_knee_angle < 185 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 181 and right_knee_angle < 185 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 180 and right_knee_angle < 184 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 180 and right_knee_angle < 184 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 179 and right_knee_angle < 183 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 169 and right_knee_angle < 173 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 127 and left_knee_angle < 131 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 186 and right_knee_angle < 190 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 183 and right_knee_angle < 187 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 185 and right_knee_angle < 189 or \
                    left_knee_angle > 182 and left_knee_angle < 186 and right_knee_angle > 187 and right_knee_angle < 191 or \
                    left_knee_angle > 194 and left_knee_angle < 198 and right_knee_angle > 201 and right_knee_angle < 205 or \
                    left_knee_angle > 202 and left_knee_angle < 206 and right_knee_angle > 210 and right_knee_angle < 214 or \
                    left_knee_angle > 197 and left_knee_angle < 201 and right_knee_angle > 206 and right_knee_angle < 210 or \
                    left_knee_angle > 198 and left_knee_angle < 202 and right_knee_angle > 207 and right_knee_angle < 211 or \
                    left_knee_angle > 196 and left_knee_angle < 200 and right_knee_angle > 208 and right_knee_angle < 212 or \
                    left_knee_angle > 197 and left_knee_angle < 201 and right_knee_angle > 207 and right_knee_angle < 211 or \
                    left_knee_angle > 196 and left_knee_angle < 200 and right_knee_angle > 206 and right_knee_angle < 210 or \
                    left_knee_angle > 194 and left_knee_angle < 198 and right_knee_angle > 203 and right_knee_angle < 207 or \
                    left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 178 and left_knee_angle < 182 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 177 and right_knee_angle < 181 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 123 and right_knee_angle < 127 or \
                    left_knee_angle > 168 and left_knee_angle < 172 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 163 and left_knee_angle < 167 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 163 and left_knee_angle < 167 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 173 and right_knee_angle < 179 or \
                    left_knee_angle > 179 and left_knee_angle < 183 and right_knee_angle > 173 and right_knee_angle < 179 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 173 and right_knee_angle < 177 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 174 and right_knee_angle < 180 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 172 and right_knee_angle < 176 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 176 and right_knee_angle < 180 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 175 and left_knee_angle < 179 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 176 and left_knee_angle < 180 and right_knee_angle > 174 and right_knee_angle < 178 or \
                    left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 175 and right_knee_angle < 179 or \
                    left_knee_angle > 174 and left_knee_angle < 178 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 173 and left_knee_angle < 177 and right_knee_angle > 171 and right_knee_angle < 175 or \
                    left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 168 and right_knee_angle < 172 or \
                    left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 160 and left_knee_angle < 190 and right_knee_angle > 170 and right_knee_angle < 190 :
                    label = "Saleum"
                    last_label = label 
    
        if left_elbow_angle > 160 and left_elbow_angle < 190 and right_elbow_angle > 160 and right_elbow_angle < 190 or \
        left_elbow_angle > 130 and left_elbow_angle < 200 and right_elbow_angle > 130 and right_elbow_angle < 200 or \
        left_elbow_angle > 130 and left_elbow_angle < 200 and right_elbow_angle > 200 and right_elbow_angle < 250 or \
        left_elbow_angle > 320 and left_elbow_angle < 360 and right_elbow_angle > 320 and right_elbow_angle < 360 or \
        left_elbow_angle > 320 and left_elbow_angle < 360 and right_elbow_angle > 20 and right_elbow_angle < 60 or \
        left_elbow_angle > 345 and left_elbow_angle < 349 and right_elbow_angle > 11 and right_elbow_angle < 15 or \
        left_elbow_angle > 246 and left_elbow_angle < 250 and right_elbow_angle > 6 and right_elbow_angle < 10 or \
        left_elbow_angle > 348 and left_elbow_angle < 352 and right_elbow_angle > 0 and right_elbow_angle < 5 or \
        left_elbow_angle > 141 and left_elbow_angle < 147 and right_elbow_angle > 324 and right_elbow_angle < 327 or \
        left_elbow_angle > 141 and left_elbow_angle <  147 and right_elbow_angle > 228 and right_elbow_angle < 232 or \
        left_elbow_angle > 153 and left_elbow_angle <  157 and right_elbow_angle > 202 and right_elbow_angle < 206 or \
        left_elbow_angle > 161 and left_elbow_angle <  165 and right_elbow_angle > 182 and right_elbow_angle < 186 or \
        left_elbow_angle > 166 and left_elbow_angle <  170 and right_elbow_angle > 177 and right_elbow_angle < 182 or \
        left_elbow_angle >  164 and left_elbow_angle < 168  and right_elbow_angle > 179 and right_elbow_angle < 183 or \
        left_elbow_angle >  242 and left_elbow_angle < 248  and right_elbow_angle > 82 and right_elbow_angle < 86 or \
        left_elbow_angle >  355 and left_elbow_angle < 359  and right_elbow_angle > 0 and right_elbow_angle < 4 or \
        left_elbow_angle >  0 and left_elbow_angle < 6  and right_elbow_angle >  337 and right_elbow_angle < 341 or \
        left_elbow_angle >  94 and left_elbow_angle < 99  and right_elbow_angle >  263 and right_elbow_angle < 267 or \
        left_elbow_angle >  142 and left_elbow_angle < 146  and right_elbow_angle >  219 and right_elbow_angle < 224 or \
        left_elbow_angle >  161 and left_elbow_angle < 164  and right_elbow_angle >  189 and right_elbow_angle < 193 or \
        left_elbow_angle >  166 and left_elbow_angle < 170  and right_elbow_angle >  172 and right_elbow_angle < 176 or \
        left_elbow_angle >  166 and left_elbow_angle < 170  and right_elbow_angle >  172 and right_elbow_angle < 176 or \
        left_elbow_angle >  165 and left_elbow_angle < 170  and right_elbow_angle >  178 and right_elbow_angle < 182 or \
        left_elbow_angle >  161 and left_elbow_angle < 165  and right_elbow_angle >  179 and right_elbow_angle < 183 or \
        left_elbow_angle >  260 and left_elbow_angle < 264  and right_elbow_angle >  81 and right_elbow_angle < 85 or \
        left_elbow_angle >  133 and left_elbow_angle < 137  and right_elbow_angle >  307 and right_elbow_angle < 311 or \
        left_elbow_angle >  149 and left_elbow_angle < 154  and right_elbow_angle >  223 and right_elbow_angle < 227 or \
        left_elbow_angle >  151 and left_elbow_angle < 157  and right_elbow_angle >  203 and right_elbow_angle < 209 or \
        left_elbow_angle >  160 and left_elbow_angle < 166  and right_elbow_angle >  177 and right_elbow_angle < 184 or \
        left_elbow_angle > 5 and left_elbow_angle < 10 and right_elbow_angle > 345 and right_elbow_angle < 349 or \
        left_elbow_angle > 20 and left_elbow_angle < 60 and right_elbow_angle > 290 and right_elbow_angle < 330:
            if left_shoulder_angle > 10 and left_shoulder_angle < 40 and right_shoulder_angle > 10 and right_shoulder_angle < 40 or \
            left_shoulder_angle > 20 and left_shoulder_angle < 60 and right_shoulder_angle > 20 and right_shoulder_angle < 60 or \
            left_shoulder_angle > 30 and left_shoulder_angle < 70 and right_shoulder_angle > 20 and right_shoulder_angle < 70 or \
            left_shoulder_angle > 90 and left_shoulder_angle < 120 and right_shoulder_angle > 90 and right_shoulder_angle < 120 or \
            left_shoulder_angle > 67 and left_shoulder_angle < 71 and right_shoulder_angle > 69 and right_shoulder_angle < 73 or \
            left_shoulder_angle > 73 and left_shoulder_angle < 77 and right_shoulder_angle > 79 and right_shoulder_angle < 83 or\
            left_shoulder_angle > 72 and left_shoulder_angle < 76 and right_shoulder_angle > 75 and right_shoulder_angle < 79 or\
            left_shoulder_angle > 59 and left_shoulder_angle < 69 and right_shoulder_angle > 72 and right_shoulder_angle < 80 or\
            left_shoulder_angle > 68 and left_shoulder_angle < 72 and right_shoulder_angle > 83 and right_shoulder_angle < 88 or\
            left_shoulder_angle > 78 and left_shoulder_angle < 82 and right_shoulder_angle > 93 and right_shoulder_angle < 97 or\
            left_shoulder_angle > 76 and left_shoulder_angle < 87 and right_shoulder_angle > 87 and right_shoulder_angle < 95 or\
            left_shoulder_angle > 63 and left_shoulder_angle < 67 and right_shoulder_angle > 70 and right_shoulder_angle < 74 or\
            left_shoulder_angle > 50 and left_shoulder_angle < 54 and right_shoulder_angle > 41 and right_shoulder_angle < 45 or\
            left_shoulder_angle > 58 and left_shoulder_angle < 69 and right_shoulder_angle > 66 and right_shoulder_angle < 72 or\
            left_shoulder_angle > 58 and left_shoulder_angle < 69 and right_shoulder_angle > 74 and right_shoulder_angle < 78 or\
            left_shoulder_angle > 75 and left_shoulder_angle < 79 and right_shoulder_angle > 92 and right_shoulder_angle < 98 or\
            left_shoulder_angle > 80 and left_shoulder_angle < 84 and right_shoulder_angle > 99 and right_shoulder_angle < 103 or\
            left_shoulder_angle > 56 and left_shoulder_angle < 60 and right_shoulder_angle > 70 and right_shoulder_angle < 74 or\
            left_shoulder_angle > 51 and left_shoulder_angle < 55 and right_shoulder_angle > 49 and right_shoulder_angle < 53 or\
            left_shoulder_angle > 56 and left_shoulder_angle < 64 and right_shoulder_angle > 68 and right_shoulder_angle < 73 or\
            left_shoulder_angle > 63 and left_shoulder_angle < 67 and right_shoulder_angle > 80 and right_shoulder_angle < 84 or\
            left_shoulder_angle > 72 and left_shoulder_angle < 76 and right_shoulder_angle > 91 and right_shoulder_angle < 96 or\
            left_shoulder_angle > 80 and left_shoulder_angle < 84 and right_shoulder_angle > 98 and right_shoulder_angle < 106 or\
            left_shoulder_angle > 74 and left_shoulder_angle < 78 and right_shoulder_angle > 90 and right_shoulder_angle < 94 or\
            left_shoulder_angle > 65 and left_shoulder_angle < 69 and right_shoulder_angle > 76 and right_shoulder_angle < 80 or\
            left_shoulder_angle > 70 and left_shoulder_angle < 90 and right_shoulder_angle > 40 and right_shoulder_angle < 80 :
                if left_knee_angle > 160 and left_knee_angle < 200 and right_knee_angle > 160 and right_knee_angle < 200 or \
                    left_knee_angle > 169 and left_knee_angle < 181 and right_knee_angle > 178 and right_knee_angle < 194 or \
                    left_knee_angle > 0 and left_knee_angle < 0 and right_knee_angle > 0 and right_knee_angle < 0 :
                    label = "Menepuk Paha" 
                    last_label = label 
    # ================================
    #  CONFIDENCE DURASI LABEL
    # ================================
    if label != last_label:
        label_duration = 0
        label_start_time = time.time()
        last_label = label
        color = (0, 255, 0)
        gesture_counter[label] += 1  # ✅ Rekap otomatis

    label_duration = round(time.time() - label_start_time, 2)

    # ================================
    # ✅ TAMPILKAN ANGKA SUDUT
    # ================================
    angles = [
        left_elbow_angle, right_elbow_angle,
        left_shoulder_angle, right_shoulder_angle,
        left_knee_angle, right_knee_angle
    ]

    angle_labels = [
        "Left Elbow", "Right Elbow",
        "Left Shoulder", "Right Shoulder",
        "Left Knee", "Right Knee"
    ]

    for i, a in enumerate(angles):
        cv2.putText(output_image, f'{angle_labels[i]}: {int(a)}',
                    (10, 60 + i * 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 0), 2)

    # ================================
    # ✅ TAMPILKAN DURASI LABEL
    # ================================
    cv2.putText(output_image, f'Durasi: {label_duration}s',
                (10, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # ================================
    # ✅ TAMPILKAN INDEX LANDMARK
    # ================================
    for i, (x, y, z) in enumerate(landmarks):
        cv2.putText(output_image, str(i), (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

    # ================================
    # ✅ TAMPILKAN LABEL
    # ================================
    cv2.putText(output_image, last_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return output_image, last_label

# Simpan rekap gerakan ke CSV
def saveGestureRecap(filename="rekap_gerakan.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file) 
        writer.writerow(["Nama Gerakan", "Jumlah Deteksi"])
        for gesture, count in gesture_counter.items():
            writer.writerow([gesture, count])

# Simpan data landmark ke CSV
def saveLandmarksToCSV(landmarks, frame_no, filename="landmarks_tepuk_murid.csv"):
    landmark_names = [name.name for name in mp_pose.PoseLandmark]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Frame", "Landmark", "X", "Y", "Z"])
        for idx, (x, y, z) in enumerate(landmarks):
            writer.writerow([frame_no, landmark_names[idx], x, y, z])
            
    saveAnglesToCSV(landmarks, frame_no)

# Simpan sudut ke CSV
def saveAnglesToCSV(landmarks, frame_no, filename="Sudut_ketrib_tepuk_murid.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "Frame",
                "Left_Elbow", "Right_Elbow",
                "Left_Shoulder", "Right_Shoulder",
                "Left_Knee", "Right_Knee"
            ])
        left_elbow = calculateAngle(landmarks[11], landmarks[13], landmarks[15])
        right_elbow = calculateAngle(landmarks[12], landmarks[14], landmarks[16])
        left_shoulder = calculateAngle(landmarks[13], landmarks[11], landmarks[23])
        right_shoulder = calculateAngle(landmarks[24], landmarks[12], landmarks[14])
        left_knee = calculateAngle(landmarks[23], landmarks[25], landmarks[27])
        right_knee = calculateAngle(landmarks[24], landmarks[26], landmarks[28])

        writer.writerow([
            frame_no,
            round(left_elbow, 2), round(right_elbow, 2),
            round(left_shoulder, 2), round(right_shoulder, 2),
            round(left_knee, 2), round(right_knee, 2)
        ])


# Fungsi utama

def main(source=0):
    mp_pose = mp.solutions.pose
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    cap = cv2.VideoCapture(source)
    cap.set(3, 1280)
    cap.set(4, 960)
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        global prev_time
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(frame, f'FPS: {int(fps)}',
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, landmarks = detectPose(frame, pose_video, display=False)

        if landmarks:
            frame, _ = classifyPose(landmarks, frame, display=False)
            saveLandmarksToCSV(landmarks, frame_count)

        cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        frame_count += 1
        cv2.imshow('Pose Classification', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    saveGestureRecap()


import tkinter as tk
from threading import Thread


def show_start_window():
    root = tk.Tk()
    root.title("Sistem Deteksi Pose")
    root.geometry("1280x720")
    root.resizable(False, False)

    background_img = Image.open("silver-dollar-eucalyptus-gray-background.jpg")
    background_img = background_img.resize((1280, 720))
    bg_photo = ImageTk.PhotoImage(background_img)

    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    big_font = font.Font(family='Helvetica', size=36, weight='bold')
    title_font = font.Font(family='Helvetica', size=48, weight='bold')

    # Judul program di atas tengah
    judul_program = "DETEKSI GERAKAN REAL-TIME PADA TARI LAWEUT MENGGUNAKAN METODE POSE LANDMARK DETECTION BASED CALCULATE ANGLE"

    title_font = font.Font(family='Helvetica', size=36 if len(judul_program) < 60 else 28, weight='bold')

    title_label = tk.Label(
        root,
        text=judul_program,
        font=title_font,
        bg="#000000",
        fg="white",
        wraplength=1000,
        justify="center"
    )

    title_label.place(relx=0.5, rely=0.1, anchor='center')

    # Tombol "Mulai" di tengah layar
    mulai_button = tk.Button(root, text="Mulai", font=big_font, bg='red', fg='white',
                             padx=40, pady=20)

    mulai_button.place(relx=0.5, rely=0.5, anchor='center')

    def tampilkan_pilihan():
        mulai_button.place_forget()
        title_label.place_forget()

        kamera_button = tk.Button(root, text="Gunakan Kamera", font=big_font, bg='blue', fg='white',
                                  padx=20, pady=10, command=pilih_kamera)
        kamera_button.place(relx=0.5, rely=0.4, anchor='center')

        video_button = tk.Button(root, text="Pilih File Video", font=big_font, bg='green', fg='white',
                                 padx=20, pady=10, command=pilih_video)
        video_button.place(relx=0.5, rely=0.6, anchor='center')

    def pilih_kamera():
        root.destroy()
        threading.Thread(target=lambda: main(0)).start()

    def pilih_video():
        filepath = filedialog.askopenfilename(
            title="Pilih Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if filepath:
            root.destroy()
            threading.Thread(target=lambda: main(filepath)).start()

    mulai_button.config(command=tampilkan_pilihan)
    root.mainloop()


if __name__ == "__main__":
    show_start_window()