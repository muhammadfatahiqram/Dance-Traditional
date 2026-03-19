import csv
import math
import cv2
import numpy as np
import mediapipe as mpexit
import mediapipe as mp
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import font, filedialog
import threading
from PIL import Image, ImageTk

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
    label = 'Nama_Gerakan_Tari'
    color = (0, 0, 255)

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        
    if left_elbow_angle > 150 and left_elbow_angle < 165 and right_elbow_angle > 40 and right_elbow_angle < 70 or \
        left_elbow_angle > 140 and left_elbow_angle < 170 and right_elbow_angle > 25 and right_elbow_angle < 50 or \
        left_elbow_angle > 140 and left_elbow_angle < 150 and right_elbow_anglsse > 30 and right_elbow_angle < 40 or \
        left_elbow_angle > 155 and left_elbow_angle < 165 and right_elbow_angle > 30 and right_elbow_angle < 40 or \
        left_elbow_angle > 155 and left_elbow_angle < 165 and right_elbow_angle > 55 and right_elbow_angle < 70 or \
        left_elbow_angle > 155 and left_elbow_angle < 165 and right_elbow_angle > 5 and right_elbow_angle < 20 or \
        left_elbow_angle > 285 and left_elbow_angle < 295 and right_elbow_angle > 190 and right_elbow_angle < 200 or \
        left_elbow_angle > 290 and left_elbow_angle < 310 and right_elbow_angle > 195 and right_elbow_angle < 210:
        if left_shoulder_angle > 20 and left_shoulder_angle < 35 and right_shoulder_angle > 40 and right_shoulder_angle < 50 or \
            left_shoulder_angle > 35 and left_shoulder_angle < 45 and right_shoulder_angle > 30 and right_shoulder_angle < 55 or\
            left_shoulder_angle > 30 and left_shoulder_angle < 45 and right_shoulder_angle > 65 and right_shoulder_angle < 80 or\
            left_shoulder_angle > 40 and left_shoulder_angle < 55 and right_shoulder_angle > 45 and right_shoulder_angle < 55 or\
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
                left_knee_angle > 335 and left_knee_angle < 350 and right_knee_angle > 170 and right_knee_angle < 180 :
                label = "Ketrib_Jaroe"    
                
    if left_elbow_angle > 125 and left_elbow_angle < 145 and right_elbow_angle > 230 and right_elbow_angle < 260 or \
        left_elbow_angle > 155 and left_elbow_angle < 170 and right_elbow_angle > 205 and right_elbow_angle < 230 or \
        left_elbow_angle > 155 and left_elbow_angle < 170 and right_elbow_angle > 190 and right_elbow_angle < 230 or \
        left_elbow_angle > 145 and left_elbow_angle < 160 and right_elbow_angle > 205 and right_elbow_angle < 230 or \
        left_elbow_angle > 145 and left_elbow_angle < 180 and right_elbow_angle > 180 and right_elbow_angle < 230 or \
        left_elbow_angle > 185 and left_elbow_angle < 210 and right_elbow_angle > 200 and right_elbow_angle < 245 or \
        left_elbow_angle > 200 and left_elbow_angle < 220 and right_elbow_angle > 270 and right_elbow_angle < 290 or \
        left_elbow_angle > 200 and left_elbow_angle < 220 and right_elbow_angle > 170 and right_elbow_angle < 200 or \
        left_elbow_angle > 200 and left_elbow_angle < 220 and right_elbow_angle > 220 and right_elbow_angle < 250 or \
        left_elbow_angle > 210 and left_elbow_angle < 240 and right_elbow_angle > 220 and right_elbow_angle < 250 or \
        left_elbow_angle > 210 and left_elbow_angle < 240 and right_elbow_angle > 195 and right_elbow_angle < 230 or \
        left_elbow_angle > 310 and left_elbow_angle < 330 and right_elbow_angle > 220 and right_elbow_angle < 240:
        if left_shoulder_angle > 10 and left_shoulder_angle < 30 and right_shoulder_angle > 10 and right_shoulder_angle < 40 or \
            left_shoulder_angle > 0 and left_shoulder_angle < 20 and right_shoulder_angle > 45 and right_shoulder_angle < 65 or \
            left_shoulder_angle > 40 and left_shoulder_angle < 60 and right_shoulder_angle > 45 and right_shoulder_angle < 65 or \
            left_shoulder_angle > 25 and left_shoulder_angle < 50 and right_shoulder_angle > 30 and right_shoulder_angle < 50 or \
            left_shoulder_angle > 40 and left_shoulder_angle < 60 and right_shoulder_angle > 60 and right_shoulder_angle < 85 or \
            left_shoulder_angle > 160 and left_shoulder_angle < 190 and right_shoulder_angle > 30 and right_shoulder_angle < 50 or \
            left_shoulder_angle > 340 and left_shoulder_angle < 360 and right_shoulder_angle > 55 and right_shoulder_angle < 75 or \
            left_shoulder_angle > 330 and left_shoulder_angle < 360 and right_shoulder_angle > 25 and right_shoulder_angle < 60 :
            if left_knee_angle > 150 and left_knee_angle < 170 and right_knee_angle > 150 and right_knee_angle < 185 or \
                left_knee_angle > 155 and left_knee_angle < 180 and right_knee_angle > 125 and right_knee_angle < 160 or \
                left_knee_angle > 140 and left_knee_angle < 155 and right_knee_angle > 100 and right_knee_angle < 130 :
                label = "Hayak_Baho"

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
            left_shoulder_angle > 320 and left_shoulder_angle < 350 and right_shoulder_angle > 15 and right_shoulder_angle < 50 :
            if left_knee_angle > 165 and left_knee_angle < 180 and right_knee_angle > 170 and right_knee_angle < 195 or \
                left_knee_angle > 160 and left_knee_angle < 190 and right_knee_angle > 90 and right_knee_angle < 130 or \
                left_knee_angle > 165 and left_knee_angle < 190 and right_knee_angle > 140 and right_knee_angle < 150 or \
                left_knee_angle > 230 and left_knee_angle < 260 and right_knee_angle > 155 and right_knee_angle < 175 or \
                left_knee_angle > 240 and left_knee_angle < 260 and right_knee_angle > 160 and right_knee_angle < 180 :
                label = "Tepuk"    
    
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
        left_elbow_angle > 0 and left_elbow_angle < 30 and right_elbow_angle > 50 and right_elbow_angle < 70:
        if left_shoulder_angle > 50 and left_shoulder_angle < 70 and right_shoulder_angle > 50 and right_shoulder_angle < 70 or \
            left_shoulder_angle > 30 and left_shoulder_angle < 50 and right_shoulder_angle > 20 and right_shoulder_angle < 40 or \
            left_shoulder_angle > 50 and left_shoulder_angle < 70 and right_shoulder_angle > 40 and right_shoulder_angle < 70 or \
            left_shoulder_angle > 20 and left_shoulder_angle < 40 and right_shoulder_angle > 0 and right_shoulder_angle < 30 or\
            left_shoulder_angle > 0 and left_shoulder_angle < 30 and right_shoulder_angle > 0 and right_shoulder_angle < 30 or\
            left_shoulder_angle > 0 and left_shoulder_angle < 30 and right_shoulder_angle > 330 and right_shoulder_angle < 360 or\
            left_shoulder_angle > 30 and left_shoulder_angle < 60 and right_shoulder_angle > 30 and right_shoulder_angle < 60 or\
            left_shoulder_angle > 5 and left_shoulder_angle < 30 and right_shoulder_angle > 20 and right_shoulder_angle < 40:
            if left_knee_angle > 170 and left_knee_angle < 200 and right_knee_angle > 170 and right_knee_angle < 200 or \
                left_knee_angle > 70 and left_knee_angle < 100 and right_knee_angle > 160 and right_knee_angle < 200 or \
                left_knee_angle > 160 and left_knee_angle < 190 and right_knee_angle > 170 and right_knee_angle < 190 :
                label = "Saleum"
    
    if left_elbow_angle > 160 and left_elbow_angle < 190 and right_elbow_angle > 160 and right_elbow_angle < 190 or \
        left_elbow_angle > 130 and left_elbow_angle < 200 and right_elbow_angle > 130 and right_elbow_angle < 200 or \
        left_elbow_angle > 130 and left_elbow_angle < 200 and right_elbow_angle > 200 and right_elbow_angle < 250 or \
        left_elbow_angle > 320 and left_elbow_angle < 360 and right_elbow_angle > 320 and right_elbow_angle < 360 or \
        left_elbow_angle > 320 and left_elbow_angle < 360 and right_elbow_angle > 20 and right_elbow_angle < 60 or \
        left_elbow_angle > 20 and left_elbow_angle < 60 and right_elbow_angle > 290 and right_elbow_angle < 330:
        if left_shoulder_angle > 10 and left_shoulder_angle < 40 and right_shoulder_angle > 10 and right_shoulder_angle < 40 or \
            left_shoulder_angle > 20 and left_shoulder_angle < 60 and right_shoulder_angle > 20 and right_shoulder_angle < 60 or \
            left_shoulder_angle > 30 and left_shoulder_angle < 70 and right_shoulder_angle > 20 and right_shoulder_angle < 70 or \
            left_shoulder_angle > 90 and left_shoulder_angle < 120 and right_shoulder_angle > 90 and right_shoulder_angle < 120 or \
            left_shoulder_angle > 70 and left_shoulder_angle < 90 and right_shoulder_angle > 40 and right_shoulder_angle < 80 :
            if left_knee_angle > 160 and left_knee_angle < 200 and right_knee_angle > 160 and right_knee_angle < 200 or \
                left_knee_angle > 0 and left_knee_angle < 0 and right_knee_angle > 0 and right_knee_angle < 0 :
                label = "Menepuk Paha" 
    
    if label != 'Nama_Gerakan_Tari':
        color = (0, 255, 0)

    # Tampilkan sudut pada layar
    angles = [left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle]
    angle_labels = ["Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder", "Left Knee", "Right Knee"]
    for i, a in enumerate(angles):
        cv2.putText(output_image, f'{angle_labels[i]} Angle: {int(a)}', (10, 60 + i * 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

    for i, (x, y, z) in enumerate(landmarks):
        cv2.putText(output_image, f'{i}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
    else:
        return output_image, label

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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while frame_count < total_frames:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
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
        
        print(f"Frame ke-{frame_count}, success={success}")

    cap.release()
    cv2.waitKey(1000)
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