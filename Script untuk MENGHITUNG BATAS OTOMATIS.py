import pandas as pd

# GANTI DENGAN NAMA FILE CSV KAMU
file_csv = "01rekap_pose.csv"

df = pd.read_csv(file_csv)

# Pastikan kolom 'label' ada
if "label" not in df.columns:
    print("❌ ERROR: Kolom 'label' tidak ditemukan di CSV!")
    print("Kolom yang tersedia:", list(df.columns))
    exit()

# Ambil semua label unik
labels = df["label"].unique()

print("\n==============================")
print(" HASIL BATAS SUDUT OTOMATIS ")
print("==============================")

for label in labels:
    print(f"\n===== BATAS SUDUT: {label} =====")
    data = df[df["label"] == label]

    for col in df.columns:
        if col not in ["label", "frame"]:
            min_val = data[col].min()
            max_val = data[col].max()

            print(f"{col:25s} : {min_val:.2f}°  -  {max_val:.2f}°")
