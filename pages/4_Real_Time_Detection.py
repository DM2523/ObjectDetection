import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode
import av
import os
import logging
# from Inference import YOLO_PRED
from ultralytics import YOLO


from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

logger = logging.getLogger(__name__)


def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers



st.set_page_config(page_title = 'WebRTC Detection',
                   layout='wide',
                   initial_sidebar_state='collapsed')

st.title('Welcome to WebRTC Video Detection App')

# with st.spinner('Loading model...'):
#     model = YOLO_PRED(onn_model='./best.onnx',data_yaml='./data.yaml')
#     # st.balloons()

with st.spinner('Loading model...'):
    model = YOLO('./best.pt')
    # st.balloons()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # img = model.predict(img)
    results = model.predict(img, conf=0.5, iou=0.5)
    
    # Get the predicted image with annotations
    predicted_img = results[0].plot()
    return av.VideoFrame.from_ndarray(predicted_img, format="bgr24")

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False})