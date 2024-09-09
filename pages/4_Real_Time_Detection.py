import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

from Inference import YOLO_PRED


st.set_page_config(page_title = 'WebRTC Detection',
                   layout='wide',
                   initial_sidebar_state='collapsed')

st.title('Welcome to WebRTC Video Detection App')

with st.spinner('Loading model...'):
    model = YOLO_PRED(onn_model='./best.onnx',data_yaml='./data.yaml')
    # st.balloons()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = model.predict(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="example", 
    video_frame_callback=video_frame_callback, 
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    log_level='debug'  # Add this for detailed logs
)


# webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False})