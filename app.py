# ===========================================
# Pose Detection & Classification (Stable)
# ===========================================
import cv2
import mediapipe as mp
import pandas as pd
import os

# ---------- Fungsi Deteksi Pose ----------
def detectPose(frame, pose, display=True):
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    landmarks = []

    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])

    if display:
        cv2.imshow("Pose Detection", frame)
    return frame, landmarks


# ---------- Fungsi Klasifikasi Pose ----------
def classifyPose(landmarks, frame, display=True):
    label = "Unknown"

    # Contoh logika sederhana
    if landmarks:
        left_shoulder = landmarks[11][1]
        right_shoulder = landmarks[12][1]
        left_hip = landmarks[23][1]
        right_hip = landmarks[24][1]

        # Berdiri tegak jika bahu dan pinggul hampir sejajar
        if abs(left_shoulder - right_shoulder) < 0.05 and abs(left_hip - right_hip) < 0.05:
            label = "Standing"
        else:
            label = "Other Pose"

    if display:
        cv2.putText(frame, f"Pose: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    return frame, label


# ---------- Fungsi Simpan Landmark ke CSV ----------
def saveLandmarksToCSV(landmarks, frame_count, folder="pose_data"):
    if not landmarks:
        return

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "pose_landmarks.csv")

    data = pd.DataFrame(landmarks, columns=["x", "y", "z", "visibility"])
    data["frame"] = frame_count

    if not os.path.exists(file_path):
        data.to_csv(file_path, index=False)
    else:
        data.to_csv(file_path, mode='a', header=False, index=False)


# ---------- Fungsi Utama ----------
def main(source=0):
    mp_pose = mp.solutions.pose
    pose_video = mp_pose.Pose(static_image_mode=False,
                              min_detection_confidence=0.5,
                              model_complexity=1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Tidak bisa membuka sumber video/kamera.")
        return

    cap.set(3, 1280)
    cap.set(4, 960)
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source != 0 else None
    print(f"🎥 Total frame video: {total_frames if total_frames else 'Kamera'}")

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Frame gagal dibaca atau video selesai.")
            break

        try:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if h == 0 or w == 0:
                continue  # skip frame rusak

            # Resize proporsional ke tinggi 640
            frame = cv2.resize(frame, (int(w * (640 / h)), 640))

            frame, landmarks = detectPose(frame, pose_video, display=False)

            if landmarks:
                frame, _ = classifyPose(landmarks, frame, display=False)
                saveLandmarksToCSV(landmarks, frame_count)

            cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            frame_count += 1

            cv2.imshow('Pose Classification', frame)

            # Tekan ESC untuk keluar
            if cv2.waitKey(1) & 0xFF == 27:
                print("🚪 ESC ditekan, keluar.")
                break

        except Exception as e:
            print(f"⚠️ Error di frame {frame_count}: {e}")
            continue

    print("✅ Proses selesai, video berakhir atau dihentikan.")
    cap.release()
    cv2.destroyAllWindows()


# ---------- Jalankan ----------
if __name__ == "__main__":
    # Ganti '0' jika ingin pakai file video, contoh:
    # main("video_test.mp4")
    main(0)
