# streamlit_app.py
from re import M
import streamlit as st

import io
from io import BytesIO
import os
import glob
import json

from PIL import Image

from grad_cam_streamlit import *


@st.cache()
def load_json_to_part_path_list(
    path: str = '/opt/ml/data/val/JPEGImages/'
    ) -> dict:

    whole_list = os.listdir(path)
    json_list = []

    for idx in whole_list:
        if idx[-4:] == 'json':
            json_list.append(idx)

    cheek_path_list = []
    forehead_path_list = [] # upper_face
    face_front_path_list = [] # mid_face
    subordinate_path_list = [] # lower_face

    cheek_list, forehead_list, face_front_list, subordinate_list = [], [], [], []

    for _json in json_list:
        with open(path + _json, 'r') as f:
            data = json.load(f)

            if data['wrinkle'] == -2 and data['hydration'] == -2:
                cheek_path_list.append(data['file_name'])
            elif data['pigmentation'] == -2 and data['hydration'] == -2:
                forehead_path_list.append(data['file_name'])
            elif data['oil'] == -2 and data['pigmentation'] == -2:
                subordinate_path_list.append(data['file_name'])
            elif data['hydration'] == -2:
                face_front_path_list.append(data['file_name'])

    for cheek_path in cheek_path_list:
        cheek_list.append(Image.open(path + cheek_path))

    for forehead_path in forehead_path_list:
        forehead_list.append(Image.open(path + forehead_path))

    for face_front_path in face_front_path_list:
        face_front_list.append(Image.open(path + face_front_path))

    for subordinate_path in subordinate_path_list:
        subordinate_list.append(Image.open(path + subordinate_path))
            
    whole_img_dict = {
        'cheek' : [cheek_path_list[:5], cheek_list[:5]],
        'forehead' : [forehead_path_list[:5], forehead_list[:5]],
        'face_front' : [face_front_path_list[:5], face_front_list[:5]],
        'subordinate' : [subordinate_path_list[:5], subordinate_list[:5]],
    }

    return whole_img_dict


def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True


if check_password():
    st.title('ArtLab X BoostCamp')
    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )

    path = '/opt/ml/data/val/JPEGImages/'

    uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
    grad_list = ['GradCAM', 'GradCAMPlusPlus', 'XGradCAM', 'EigenCAM', 'LayerCAM', 'EigenGradCAM']

    st.sidebar.header('Main')
    select_function = st.sidebar.selectbox('Select Function', ['XAI', 'Grad CAM'])
    
    if select_function == 'Grad CAM':
        st.sidebar.header('Grad CAM')
        load_grad_model = st.sidebar.selectbox('Select Model', grad_list)

        whole_img_dict = load_json_to_part_path_list()

        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))

            st.image(image, caption='Uploaded Image')

            recipe_button = st.button('Show GradCam🚀')

            if recipe_button:
                st.markdown(
                "<hr />",
                unsafe_allow_html=True
                )
                gradcam = gradcam = show_output(uploaded_file, load_grad_model)
                st.image(gradcam)
        else:
            # Sidebar Menu
            st.sidebar.header('Sidebar Menu')
            load_part = st.sidebar.selectbox('View by part', ['forehead', 'face front', 'cheek', 'subordinate'])

            if load_part == 'forehead':
                img_path = st.sidebar.selectbox('Image Name', whole_img_dict['forehead'][0])
            elif load_part == 'face front':
                img_path = st.sidebar.selectbox('Image Name', whole_img_dict['face_front'][0])
            elif load_part == 'cheek':
                img_path = st.sidebar.selectbox('Image Name', whole_img_dict['cheek'][0])
            elif load_part == 'subordinate':
                img_path = st.sidebar.selectbox('Image Name', whole_img_dict['subordinate'][0])

            img = Image.open(path + img_path)

            st.image(img, caption='Uploaded Image')

            recipe_button = st.button('Show GradCam🚀')
            
            if recipe_button:
                st.markdown(
                "<hr />",
                unsafe_allow_html=True
                )
                gradcam = show_output(path + img_path, load_grad_model)
                st.image(gradcam, caption='Image with Gradcam')
    elif select_function == 'XAI':
        pass
