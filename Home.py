import streamlit as st

st.set_page_config(page_title = 'Home',
                   layout='wide',
                   initial_sidebar_state='auto')
st.title('YOLOV8 Object Detection App')
st.caption('This webapp uses fine tuned yolov8s.pt to detect objects.')

st.markdown("""
### This App detects objects from Images and Real Time feed.
-20 Different objects can be detected.\n
[click here for images](/Detect_Images) &nbsp;&nbsp; [click here for WebRTC](/Real_Time_Detection)\n
            
Below are the objects that out model can detect:
            
1. Person
2. Car
3. Chair
4. Bottle
5. Potted Plant
6. Bird
7. Dog
8. Sofa
9. Bicycle
10. Horse
11. Boat
12. Motorbike
13. Cat
14. Tvmonitor
15. Cow
16. Sheep
17. Aeroplane
18. Train
19. Diningtable
20. Bus

""")