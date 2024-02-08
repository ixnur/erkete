import cv2
import numpy as np
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO
import time

LED_PIN = 18  #değiştirilebilir
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

model = load_model('egitilmis_model.h5')

class_labels = ['silah', 'kesici alet', 'tabanca', 'tornavida', 'taş', 'bıçak', 'çakı', 'levye']
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    predictions = model.predict(input_frame)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    if predicted_class in ['silah', 'kesici alet', 'tabanca']:
        cv2.putText(frame, f'Dikkat! {predicted_class} algılandı.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        GPIO.output(LED_PIN, GPIO.HIGH)
    else:
        GPIO.output(LED_PIN, GPIO.LOW)
    
    #ekranda göster
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
