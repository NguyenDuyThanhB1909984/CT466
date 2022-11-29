import streamlit as st
import cv2
import numpy as np
import detect2
import torch


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
# model2 = torch.hub.load('WongKinYiu/yolov7', 'custom', path='yolov7.pt')
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='bestfinal.pt')
img_file_buffer = st.camera_input("Take a picture")

#plate = 0 
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    #st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(cv2_img.shape)

    plate, img, crop = detect2.plate(cv2_img,model, model2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(crop,"nhan dang ky tu")
    with col2:
        # st.write("Bien so la: ", plate[0],plate[1],"-",plate[2],plate[3],plate[4],plate[5],plate[6],".",plate[7],plate[8])
        st.write(plate)
    