import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

#create the window
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("400x300")

#Load face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Load recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#global variables
dataset_path = 'dataset'
attendance = []
user_info_file = 'user_info.csv'

#create dataset directory if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

#create user info CSV if it doesn't exist
if not os.path.exists(user_info_file):
    pd.DataFrame(columns=["ID", "Name"]).to_csv(user_info_file, index=False)


#capturing the faces

def capture_faces():
    name = name_entry.get().strip()
    person_id = person_id_entry.get().strip()

    if not name or not person_id.isdigit():
        messagebox.showerror("Error", "Please enter a valid name and numeric ID.")
        return

    person_id = int(person_id)

    # Save ID-name to CSV if not already present
    user_info = pd.read_csv(user_info_file)
    if person_id not in user_info["ID"].values:
        # user_info = user_info.append({"ID": person_id, "Name": name}, ignore_index=True)
        # user_info.to_csv(user_info_file, index=False)
        new_row = pd.DataFrame([{"ID": person_id, "Name": name}])
        user_info = pd.concat([user_info, new_row], ignore_index=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            filename = f"{dataset_path}/{name.lower().replace(' ', '_')}.{person_id}.{count}.jpg"
            cv2.imwrite(filename, face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {count}/30", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Capturing Faces', frame)

        if cv2.waitKey(1) == ord('q') or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Captured {count} images for {name}")


#function to train the recognizer

def train_model():
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    faces = []
    ids = []

    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # id = int(os.path.split(image_path)[-1].split('.')[1])
        id = int(os.path.split(image_path)[-1].split('.')[-3])

        faces.append(gray_img)
        ids.append(id)

    if not faces:
        messagebox.showerror("Error", "No training data found. Capture faces first!")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')
    messagebox.showinfo("Info", "Model trained and saved as 'trainer.yml'")


#to recognise faces

def recognize_faces():
    try:
        recognizer.read('trainer.yml')
    except:
        messagebox.showerror("Error", "Train the model first!")
        return

    # Load ID-name mapping
    if not os.path.exists(user_info_file):
        messagebox.showerror("Error", "No user info found.")
        return

    user_info = pd.read_csv(user_info_file)
    names = dict(zip(user_info["ID"], user_info["Name"]))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(face)

            if conf < 70:
                name = names.get(id_, "Unknown")
                if name != "Unknown" and name not in [entry["Name"] for entry in attendance]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance.append({"Name": name, "Time": timestamp})
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save attendance
    if attendance:
        df = pd.DataFrame(attendance)
        # df.to_csv("attendance.csv", index=False)   ----depreciated
        file_exists = os.path.exists("attendance.csv")
        df.to_csv("attendance.csv", mode='a', index=False, header=not file_exists)

        messagebox.showinfo("Info", "Attendance saved to 'attendance.csv'")
    else:
        messagebox.showinfo("Info", "No attendance to save.")


#the GUI
name_label = tk.Label(root, text="Name:")
name_label.pack(pady=5)
name_entry = tk.Entry(root)
name_entry.pack(pady=5)

person_id_label = tk.Label(root, text="ID (Numeric):")
person_id_label.pack(pady=5)
person_id_entry = tk.Entry(root)
person_id_entry.pack(pady=5)

capture_button = tk.Button(root, text="Capture Faces", width=20, command=capture_faces)
capture_button.pack(pady=10)

train_button = tk.Button(root, text="Train Model", width=20, command=train_model)
train_button.pack(pady=10)

recognize_button = tk.Button(root, text="Start Recognition", width=20, command=recognize_faces)
recognize_button.pack(pady=10)

# -----------------------------
# Run the GUI App
# -----------------------------
root.mainloop()
