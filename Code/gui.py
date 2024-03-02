import tkinter as tk
from tkinter.ttk import *
from tkinter import ttk
import numpy as np
import cv2
import pickle
from PIL import Image, ImageTk

face_cascade = cv2.CascadeClassifier('assets\haarcascades\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

def mark_attendance(name):
    # Add your attendance marking code here
    attendance_label.config(text="Attendance Marked for " + name)

def detect_faces():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 :
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 3
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            mark_attendance(name)
        img_item = "test.png"
        cv2.imwrite(img_item, roi_color)
        color = (255, 0, 0)
        stroke = 4
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize((400, 400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    live_stream_label.config(image=img)
    live_stream_label.image = img # keep a reference to avoid garbage collection
    root.after(10, detect_faces)

def capture_frame():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 :
            name = labels[id_]
            mark_attendance(name)
            img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
            img = img.resize((400, 400), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            captured_image_label.config(image=img)
            captured_image_label.image = img # keep a reference to avoid garbage collection
            captured_image_label.attendance_name = name
    root.after(5000, capture_frame)

def capture_button():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor =1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 :
            name = labels[id_]
            mark_attendance(name)
            img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
            img = img.resize((400, 400), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            captured_image_label.config(image=img)
            captured_image_label.image = img # keep a reference to avoid garbage collection
            captured_image_label.attendance_name = name
            img_item = "test.png"
            cv2.imwrite(img_item, roi_color)
            attendance_label.config(text="Attendance Marked for " + captured_image_label.attendance_name)

root = tk.Tk()
root.title("Attendance System")

live_stream_label = tk.Label(root)
live_stream_label.pack(side=tk.LEFT)

captured_image_frame = tk.Frame(root)
captured_image_frame.pack(side=tk.RIGHT)

captured_image_label = tk.Label(captured_image_frame)
captured_image_label.pack(side=tk.TOP)

attendance_label = tk.Label(captured_image_frame, font=("Courier", 16))
attendance_label.pack(side=tk.BOTTOM)
style = ttk.Style()
style.configure('Custom.TButton', font=('Helvetica', 12), foreground='black', background='#4CAF50', width=10)
capture_button = ttk.Button(captured_image_frame, text="Capture", command=capture_button, style='Custom.TButton')
capture_button.pack(side=tk.BOTTOM, pady = 20)
capture_button.configure(style='Custom.TButton')

detect_faces()
root.mainloop()
