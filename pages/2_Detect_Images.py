import streamlit as st
from Inference import YOLO_PRED
from PIL import Image
import numpy as np

st.set_page_config(page_title = 'Image Detection',
                   layout='wide',
                   initial_sidebar_state='collapsed')

st.title('Welcome to Image Detection App')
st.caption('Please upload image to get detections.')

with st.spinner('Loading model...'):
    model = YOLO_PRED(onn_model='./best.onnx',data_yaml='./data.yaml')
    # st.balloons()


def upload_image():
    # Upload image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": "{:,.2f}MB".format(image_file.size / 1024**2)
        }

        # st.write(file_details)

        # Validate File
        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('VALID FILE.')
            return image_file, file_details
        else:
            st.error('INVALID FILE. Please upload a valid image file.')
            return None, None
    return None, None


def main():
    image_file,file_details = upload_image()
    if image_file:
        prediction = False
        image  = Image.open(image_file)

        col1, col2 = st.columns(2)
        with col1:
            st.info('Preview of uploaded image')
            st.image(image_file,use_column_width=True)

        with col2:
            st.subheader('Check file details')
            st.json(file_details)

            button = st.button('GET DETECTIONS')
            # image_arr = np.array(image)
            # st.write("Image array shape:", image_arr.shape)


            if button:
                with st.spinner('Predicting...'):
                    image_arr = np.array(image)
                    pred_img = model.predict(image_arr)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True

        if prediction:
            # Add border to the image
            st.subheader('Detected Objects from the model.')
            st.image(pred_img_obj)




if __name__ == '__main__':
    main()


