import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
from streamlit_drawable_canvas import st_canvas

#Model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#load the model
model = MyModel()
model.load_state_dict(torch.load("C:\\Users\\username\\....pth", map_location=torch.device('cpu')))
#Change the file path yourself, where it will be your pytorch file xxx.pth
model.eval() #set evaluation status

#these add text to the frontend
st.title('My MNIST model')
st.markdown('''
Write a digit range from 0 to 9! 
Press <Predict> to see the prediction!
''')


SIZE = 192 #Size for drawing area

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE, #square drawing area
    drawing_mode="freedraw",
    key='canvas')


#process the image only if there is drawing
if np.any(canvas_result.image_data):
    #resize the 192x192 canvas drawing to 28x28
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    #display the resized image (if you dont resize it back to 192x192, it'll show up as a small image)
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    st.write('Model Input')
    st.image(rescaled)

# Predict button is pressed
if st.button('Predict'):
    # Convert image from colour to gray
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Change the values from [0,255] to [0,1] then convert the numpy array to a torch tensor
    bwx = torch.from_numpy(test_x.reshape(1, 28, 28)/255)
    # Change input to float from double then unsqueeze changes from (1,28,28) to (1,1,28,28)
    val = model(bwx.float().unsqueeze(0))
    # Apply softmax to the output to get probabilities
    probabilities = torch.nn.functional.softmax(val, dim=1)
    st.write(f'result: {torch.argmax(probabilities).item()}')
    #Show the probabilities predicted in bar chart form
    st.bar_chart(probabilities.detach().numpy()[0])
