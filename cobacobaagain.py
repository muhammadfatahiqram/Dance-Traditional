import csv
import os
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
        left_elbow_angle > 331 and left_elbow_angle < 335 and right_elbow_angle > 175 and right_elbow_angle < 179 or \
        left_elbow_angle > 290 and left_elbow_angle < 310 and right_elbow_angle > 195 and right_elbow_angle < 210:
        if left_shoulder_angle > 20 and left_shoulder_angle < 35 and right_shoulder_angle > 40 and right_shoulder_angle < 50 or \
            left_shoulder_angle > 35 and left_shoulder_angle < 45 and right_shoulder_angle > 30 and right_shoulder_angle < 55 or\
            left_shoulder_angle > 33  and left_shoulder_angle < 37  and right_shoulder_angle > 77  and right_shoulder_angle < 81  or \
            left_shoulder_angle > 55 and left_shoulder_angle < 90 and right_shoulder_angle > 45 and right_shoulder_angle < 65 :
            if left_knee_angle > 110 and left_knee_angle < 120 and right_knee_angle > 175 and right_knee_angle < 185 or \
                left_knee_angle > 0 and left_knee_angle < 20 and right_knee_angle > 170 and right_knee_angle < 185 or \
                left_knee_angle > 177 and left_knee_angle < 181 and right_knee_angle > 177 and right_knee_angle < 181 or \
                left_knee_angle > 335 and left_knee_angle < 350 and right_knee_angle > 170 and right_knee_angle < 180 :
                label = "Ketrib_Jaroe"   
                last_label = label 

        if left_elbow_angle > 125 and left_elbow_angle < 145 and right_elbow_angle > 230 and right_elbow_angle < 260 or \
        left_elbow_angle > 120 and left_elbow_angle < 170 and right_elbow_angle > 205 and right_elbow_angle < 240 or \
        left_elbow_angle > 153 and left_elbow_angle < 163 and right_elbow_angle > 190 and right_elbow_angle < 218 or \
        left_elbow_angle > 310 and left_elbow_angle < 330 and right_elbow_angle > 220 and right_elbow_angle < 240:
            if left_shoulder_angle > 10 and left_shoulder_angle < 30 and right_shoulder_angle > 10 and right_shoulder_angle < 40 or \
                left_shoulder_angle > 0 and left_shoulder_angle < 20 and right_shoulder_angle > 45 and right_shoulder_angle < 65 or \
                left_shoulder_angle > 45 and left_shoulder_angle < 70 and right_shoulder_angle > 39 and right_shoulder_angle < 66 or \
                left_shoulder_angle > 330 and left_shoulder_angle < 360 and right_shoulder_angle > 25 and right_shoulder_angle < 60 :
                if left_knee_angle > 150 and left_knee_angle < 170 and right_knee_angle > 150 and right_knee_angle < 185 or \
                    left_knee_angle > 156 and left_knee_angle < 163 and right_knee_angle > 141 and right_knee_angle <  150 or \
                    left_knee_angle > 140 and left_knee_angle < 155 and right_knee_angle > 100 and right_knee_angle < 130 :
                    label = "Hayak_Baho"
                    last_label = label

    if left_elbow_angle > 160 and left_elbow_angle < 180 and right_elbow_angle > 290 and right_elbow_angle < 310 or \
        left_elbow_angle > 166 and left_elbow_angle < 177 and right_elbow_angle > 176 and right_elbow_angle < 190 or \
        left_elbow_angle > 110 and left_elbow_angle < 130 and right_elbow_angle > 280 and right_elbow_angle < 300:
        if left_shoulder_angle > 10 and left_shoulder_angle < 30 and right_shoulder_angle > 55 and right_shoulder_angle < 70 or \
                left_knee_angle > 164 and left_knee_angle < 170 and right_knee_angle > 132 and right_knee_angle < 136 or \
                left_knee_angle > 240 and left_knee_angle < 260 and right_knee_angle > 160 and right_knee_angle < 180 :
                label = "Tepuk"    
                last_label = label 
    
        if left_elbow_angle > 120 and left_elbow_angle < 140 and right_elbow_angle > 250 and right_elbow_angle < 280 or \
            left_elbow_angle > 257 and left_elbow_angle < 261 and right_elbow_angle > 135 and right_elbow_angle < 139 or \
            left_elbow_angle > 0 and left_elbow_angle < 30 and right_elbow_angle > 50 and right_elbow_angle < 70:
            if left_shoulder_angle > 50 and left_shoulder_angle < 70 and right_shoulder_angle > 50 and right_shoulder_angle < 70 or \
                left_shoulder_angle > 30 and left_shoulder_angle < 50 and right_shoulder_angle > 20 and right_shoulder_angle < 40 or \
                left_shoulder_angle > 0   and left_shoulder_angle < 4   and right_shoulder_angle > 5   and right_shoulder_angle < 9   or \
                left_shoulder_angle > 5 and left_shoulder_angle < 30 and right_shoulder_angle > 20 and right_shoulder_angle < 40:
                if left_knee_angle > 170 and left_knee_angle < 200 and right_knee_angle > 170 and right_knee_angle < 200 or \
                    left_knee_angle > 70 and left_knee_angle < 100 and right_knee_angle > 160 and right_knee_angle < 200 or \
                    left_knee_angle > 171 and left_knee_angle < 175 and right_knee_angle > 168 and right_knee_angle < 172 or \
                    left_knee_angle > 169 and left_knee_angle < 173 and right_knee_angle > 170 and right_knee_angle < 174 or \
                    left_knee_angle > 160 and left_knee_angle < 190 and right_knee_angle > 170 and right_knee_angle < 190 :
                    label = "Saleum"
                    last_label = label 
    
        if left_elbow_angle > 160 and left_elbow_angle < 190 and right_elbow_angle > 160 and right_elbow_angle < 190 or \
        left_elbow_angle > 130 and left_elbow_angle < 200 and right_elbow_angle > 130 and right_elbow_angle < 200 or \
        left_elbow_angle >  160 and left_elbow_angle < 166  and right_elbow_angle >  177 and right_elbow_angle < 184 or \
        left_elbow_angle > 5 and left_elbow_angle < 10 and right_elbow_angle > 345 and right_elbow_angle < 349 or \
        left_elbow_angle > 20 and left_elbow_angle < 60 and right_elbow_angle > 290 and right_elbow_angle < 330:
            if left_shoulder_angle > 10 and left_shoulder_angle < 40 and right_shoulder_angle > 10 and right_shoulder_angle < 40 or \
            left_shoulder_angle > 20 and left_shoulder_angle < 60 and right_shoulder_angle > 20 and right_shoulder_angle < 60 or \
            left_shoulder_angle > 65 and left_shoulder_angle < 69 and right_shoulder_angle > 76 and right_shoulder_angle < 80 or\
            left_shoulder_angle > 70 and left_shoulder_angle < 90 and right_shoulder_angle > 40 and right_shoulder_angle < 80 :
                if left_knee_angle > 160 and left_knee_angle < 200 and right_knee_angle > 160 and right_knee_angle < 200 or \
                    left_knee_angle > 169 and left_knee_angle < 181 and right_knee_angle > 178 and right_knee_angle < 194 or \
                    left_knee_angle > 0 and left_knee_angle < 0 and right_knee_angle > 0 and right_knee_angle < 0 :
                    label = "Menepuk Paha" 
                    last_label = label 
    # ================================
    # ✅ CONFIDENCE DURASI LABEL
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

    return (output_image, last_label,
        left_elbow_angle, right_elbow_angle,
        left_shoulder_angle, right_shoulder_angle,
        left_knee_angle, right_knee_angle)


# Simpan data landmark ke CSV
csv_file = "000rekap_pose.csv"

def saveLandmarksToCSV(frame_number, label,
                       left_elbow_angle, right_elbow_angle,
                       left_shoulder_angle, right_shoulder_angle,
                       left_knee_angle, right_knee_angle):

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "frame", "label",
                "left_elbow_angle", "right_elbow_angle",
                "left_shoulder_angle", "right_shoulder_angle",
                "left_knee_angle", "right_knee_angle"
            ])

        writer.writerow([
            frame_number, label,
            left_elbow_angle, right_elbow_angle,
            left_shoulder_angle, right_shoulder_angle,
            left_knee_angle, right_knee_angle
        ])

# Simpan sudut ke CSV
def saveAnglesToCSV(frame_no, label,
                     left_elbow, right_elbow,
                     left_shoulder, right_shoulder,
                     left_knee, right_knee,
                     filename="Sudut_ketrib_tepuk_murid.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Frame", "Label",
                "Left_Elbow", "Right_Elbow",
                "Left_Shoulder", "Right_Shoulder",
                "Left_Knee", "Right_Knee"
            ])

        writer.writerow([
            frame_no, label,
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
            (frame, label,
            left_elbow_angle, right_elbow_angle,
            left_shoulder_angle, right_shoulder_angle,
            left_knee_angle, right_knee_angle) = classifyPose(landmarks, frame, display=False)

            saveAnglesToCSV(
                frame_count, label,
                left_elbow_angle, right_elbow_angle,
                left_shoulder_angle, right_shoulder_angle,
                left_knee_angle, right_knee_angle
)

        cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        frame_count += 1
        cv2.imshow('Pose Classification', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


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