import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tensorflow as tf 
from PIL import Image, ImageTk
import numpy as np
import cv2

from tensorflow.keras.models import model_from_json
#from tensorflow.keras.models import model_from_json
#from tensorflow import keras
#model = keras.models.model_from_json(...)  # ... rest of your code

def FacialExpressionModel(json_file,weights_file):
    with open(json_file,"r")as file:
        loaded_model_json=file.read()
        model=model_from_json(loaded_model_json)
        
    model.load_weights(weights_file) 
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) 
    return model 
top=tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1=Label(top,background='#CDCDCD',font=('arial,15,bold'))
sign_image=Label(top)
facec = cv2.CascadeClassifier(r"C:\Users\ASUS\Desktop\Emotion detection p\haarcascade_frontalface_default.xml")
model = FacialExpressionModel(r"C:\Users\ASUS\Desktop\Emotion detection p\model.json", r"C:\Users\ASUS\Desktop\Emotion detection p\val_accuracy.weights.h5")

EMOTIONS_LIST=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def Detect(file_path):
    global label_packed
    
    image=cv2.imread(file_path)
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=facec.detectMultiScale(gray_image,1.3,5)
    try:
        for(x,y,w,h)in faces:
            fc=gray_image[y:y+h,x:x+w]
            roi=cv2.resize(fc,(48,48))
            pred= EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print('Predicted Emotion is'+pred )
        label1.configure(foreground='#011638',text=pred)
    except:
        label1.configure(foreground='#011638',text='unable to detect ')
        
def show_detect_button(file_path):
    detect_b=Button(top,text="Detect Emotion",command=lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.46)
    
'''def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.3,(top.winfo_height()/2.3))))
        im= ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        show_detect_button(file_path)
    except:
        pass
        
# Upload image function with larger display
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        # Resize the image for better visibility in the UI
        uploaded.thumbnail((300, 300))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')  # Clear the previous emotion text
        show_detect_button(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load image: {e}") ''' 
        
def upload_image():
    """
    Uploads an image, resizes it for better UI display, and updates the label.
    Handles potential errors during image loading and provides a user-friendly
    error message.  """
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        # Resize the image to a fixed size (300x300 in this example)
        uploaded.thumbnail((300, 300))
        # Convert the image to PhotoImage format for Tkinter display
        im = ImageTk.PhotoImage(uploaded)

        # Update the label with the resized image
        sign_image.configure(image=im)
        sign_image.image = im  # Keep a reference to avoid garbage collection

        # Clear the previous emotion text (optional)
        label1.configure(text='')

        show_detect_button(file_path)

    except Exception as e:
        messagebox.showerror("Error", f"Could not load image: {e}")   
        
             
    
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)  # Adjust this line's indentation if needed 
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'))  # Dark blue background, white text
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom',expand='True')
heading=Label(top,text='Emotion Detection',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()        