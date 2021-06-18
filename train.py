import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


train_dir = 'data/train' #Dette er stien til vores data/train mappe
val_dir = 'data/test' #Dette er stien til vores data/test mappe

# Vores ImageDataGenerator udvider størrelsen af vores datasæt,
# ved at generere modifiserede versioner af hvert billed der er i datamappen.
train_datagen = ImageDataGenerator(rescale=1./255) 
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48), 
    batch_size=64, 
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')


# Vi bygger her vores CNN (Convolutional Neural Network)
network_model = Sequential()

network_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1))) #Conv2D produce a tensor of outputs 
network_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
network_model.add(MaxPooling2D(pool_size=(2, 2))) # Max2D downsampler
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



# Vi compiler og træner modellen. Istedet for at bruge SGD (Stochastic gradient descent),
# bruger vi "Adam" som vores optimizer. Adam er en optimization algoritme.
# Adam giver bedre resultater og er hurtigere end SGD.
network_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])
network_model_info = network_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64)

network_model.save_weights('trained_face_model.h5')


# 1  loss: 1.8270 - accuracy: 0.2433 - val_loss: 1.6966 - val_accuracy: 0.3383
# 25 loss: 0.8402 - accuracy: 0.6913 - val_loss: 1.0854 - val_accuracy: 0.5995
# 50 loss: 0.4022 - accuracy: 0.8579 - val_loss: 1.1834 - val_accuracy: 0.6133

#Focus noise 
#Improves oerformance, hurts accuracy 

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    face_cascade = cv2.CascadeClassifier('C:/Users/danie/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces: 
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) #(Image, Start_point, End_point, Color, Thickness)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims( cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = network_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, emotion_dict[maxindex], (x+20,y-60), font, 1, (255, 255, 255), 2, cv2.LINE_AA) #(Image, emotion_dict, coordinates, text type, text størrelse, farve, tykkelse, linjetype)
    cv2.imshow('Video', cv2.resize( frame, (1200, 860), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        capture.release()
        cv2.destroyAllWindows()
        break
