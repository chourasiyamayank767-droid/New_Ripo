import cv2
import os
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH = "dataset"
MODEL_FILE = "model.pkl"
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ========= CAMERA FIX =========
def get_camera():
    print("Searching for camera...")

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        for i in range(5):
            cap = cv2.VideoCapture(i, backend)

            if cap is not None and cap.isOpened():
                print(f"✅ Camera found at index {i}")
                return cap

            cap.release()

    print("❌ No camera detected!")
    return None


# ================= DATA COLLECTION =================
def collect_data():
    name = input("Enter your name: ").strip()

    if name == "":
        print("Name cannot be empty!")
        return

    person_path = os.path.join(DATASET_PATH, name)
    os.makedirs(person_path, exist_ok=True)

    cap = get_camera()

    if cap is None:
        print("❌ No camera found!")
        return

    count = 0
    print("Collecting images...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera error!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (100, 100))

            cv2.imwrite(
                os.path.join(person_path, f"{count}.jpg"),
                face
            )

            count += 1

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Data collection done!")


# ================= TRAIN MODEL =================
def train_model():
    faces = []
    labels = []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = cv2.resize(img, (100, 100))

            faces.append(img.flatten())
            labels.append(person)

    if len(faces) == 0:
        print("No data found! Collect dataset first.")
        return

    faces = np.array(faces)

    print("Training model...")

    n_components = min(50, len(faces))

    pca = PCA(n_components=n_components)

    faces_pca = pca.fit_transform(faces)

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(faces_pca, labels)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((pca, knn), f)

    print("Model trained successfully!")


# ================= ATTENDANCE =================
def mark_attendance(name):

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)

    else:
        df = pd.DataFrame(
            columns=["Name", "Time"]
        )

    if name not in df["Name"].values:

        now = datetime.now()

        df.loc[len(df)] = [
            name,
            now.strftime("%H:%M:%S")
        ]

        df.to_csv(
            ATTENDANCE_FILE,
            index=False
        )


def start_attendance():

    if not os.path.exists(MODEL_FILE):
        print("Train model first!")
        return

    with open(MODEL_FILE, "rb") as f:
        pca, knn = pickle.load(f)

    cap = get_camera()

    if cap is None:
        print("❌ No camera found!")
        return

    print("Press ESC to exit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera error!")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5
        )

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            try:
                face = cv2.resize(
                    face,
                    (100, 100)
                ).flatten().reshape(1, -1)

                face_pca = pca.transform(face)

                name = knn.predict(face_pca)[0]

            except:
                name = "Unknown"

            mark_attendance(name)

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Attendance",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ================= MENU =================
while True:

    print("\n===== FACE ATTENDANCE SYSTEM =====")

    print("1. Collect Dataset")
    print("2. Train Model")
    print("3. Start Attendance")
    print("4. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        collect_data()

    elif choice == "2":
        train_model()

    elif choice == "3":
        start_attendance()

    elif choice == "4":
        break

    else:
        print("Invalid choice!")





































