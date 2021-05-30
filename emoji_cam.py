import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import os
import threading

from fastapi import FastAPI

app = FastAPI()



network_model = Sequential()

network_model.add(Conv2D(32, kernel_size=(  3, 3), activation='relu', input_shape=(48, 48, 1)))
network_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
network_model.add(MaxPooling2D(pool_size=(2, 2)))
network_model.add(Dropout(0.25))
network_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
network_model.add(MaxPooling2D(pool_size=(2, 2)))
network_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
network_model.add(MaxPooling2D(pool_size=(2, 2)))
network_model.add(Dropout(0.25))
network_model.add(Flatten())
network_model.add(Dense(1024, activation='relu'))
network_model.add(Dropout(0.5))
network_model.add(Dense(7, activation='softmax'))

network_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dict = {0:cur_path +"/emoji/angry.png",2:cur_path +"/emoji/disgusted.png",2:cur_path +"/emoji/fearful.png",3:cur_path +"/emoji/happy.png",4:cur_path +"/emoji/neutral.png",5:cur_path +"/emoji/sad.png",6:cur_path +"/emoji/surpriced.png"}
#emoji_dict = ["/emoji/angry.png", "/emoji/disgusted.png", "/emoji/fearful.png", "/emoji/happy.png", "/emoji/neutral.png", "/emoji/sad.png", "/emoji/surpriced.png"]
global frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)
global capture
show_text = [0]

def show_camera():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        print("cant open the camera")
    #global frame_number
    #length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    #frame_number += 1
    #if frame_number >= length:
    #   exit()
    #cap1.set(1, frame_number)
    ret, camera_frame = capture.read() #ret returns true if camera is available
    camera_frame = cv2.resize(camera_frame, (600, 500))
    bounding_box = cv2.CascadeClassifier('C:/Users/danie/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(camera_frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = network_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(camera_frame, emotion_dict[maxindex], (x+20, y-60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if ret is None:
        print("Major error!")
    elif ret:
        global frame
        frame = camera_frame.copy()
        pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
        root.update()
        camera_label.after(10, show_camera)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def show_emoji():
    emoji_frame = cv2.imread(emoji_dict[show_text[0]])
    img2 = Image.fromarray(emoji_frame)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    emoji_label.imgtk2 = imgtk2
    emoji_text.configure(text=emotion_dict[show_text[0]], font=('arial', 40, 'bold'))
    emoji_label.configure(image=imgtk2)
    emoji_label.after(10, show_emoji)
    
    
if __name__ == '__main__':
    root = tk.Tk()
    heading2 = Label(root, text="EMOJI CAM", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
    heading2.pack()
    camera_label = tk.Label(master=root, padx=50, bd=10)
    emoji_label = tk.Label(master=root, bd=10)
    emoji_text = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    camera_label.pack(side=LEFT)
    camera_label.place(x=100, y=300)
    emoji_text.pack()
    emoji_text.place(x=1300, y=180)
    emoji_label.pack(side=RIGHT)
    emoji_label.place(x=1100, y=300)
    root.title("EMOJI CAM")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)
    #show_camera()
    #show_emoji()
    threading.Thread(target=show_camera).start()
    threading.Thread(target=show_emoji).start()
    root.mainloop()
