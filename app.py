import queue
import sqlite3

import PIL.Image
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from src.api import load, get_model

conn = sqlite3.connect('database_colddrinks.db')
c = conn.cursor()
Flag = 0

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS objectcount(item TEXT,itemcount INT,postdate DATE)')


create_table()


def add_data(item, itemcount, postdate):
    c.execute('INSERT INTO objectcount(item,itemcount,postdate) VALUES (?,?,?)',
              (item, itemcount, postdate))
    conn.commit()


def view_all():
    c.execute('SELECT * FROM objectcount')
    data = c.fetchall()
    return data


def csvformat(data):
    df = pd.DataFrame(data)
    df.index.name = 'Ser No'
    df.to_csv('cold_drinks.csv')
    #st.write('Data is written successfully to csv File.') 

def excelformat(data):
    df = pd.DataFrame(data)
    #df.to_excel('wine.xlsx')
    # create excel writer object
    df.index.name = 'Ser No'
    writer = pd.ExcelWriter('cold_drinks.xlsx')
    # write dataframe to excel
    df.to_excel(writer)
    # save the excel
    writer.save()


class VideoProcessor(VideoProcessorBase):
    """
    class for taking frame / sec and predict on it
    """

    def __init__(self):
        self.confidence_threshold = CONF_THR
        self.result_queue = queue.Queue()
        self.type = None

    def recv(self, frame):
        """
        :param frame: image array (height, width, channel)
        :return: image (height, width, channel) with bounding box
        """
        image_ = frame.to_ndarray(format="bgr24")
        # if self.type == "yes":
        img, counting_ = load(model, image_, self.confidence_threshold, IMAGE_SIZE)
        C_ = {k: v for k, v in counting_.items() if v > 0}
        self.result_queue.put([C_])
        # elif self.type == "no":
        #     self.result_queue.put([None])
        return av.VideoFrame.from_ndarray(img, format="bgr24")


@st.cache
def __model():
    """
    load model in cache mode
    Returns: torch.Module
    """
    # return get_model('best3.pt')
    return get_model('exp2/weights/best.pt')   


head_message_temp = """
<div style="padding:10px;border-radius:5px;margin:10px;">
<h3 style="text-align:center;">{}</h3>
<h3>Count: {}</h6> 
<h3>Date : {}<h3>
</div>
"""

if __name__ == '__main__':

    st.title("Cold Drinks Inventory Management System")
    style = """<style>
    footer {visibility: hidden;}
                            </style>
                            """
    WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
    )  # permission for camera
    st.markdown(style, unsafe_allow_html=True)
    # mode_ = st.sidebar.radio(
    #     "Mode", ('Staff', 'Admin'))
    mode_ = st.selectbox('Mode', ('None','Staff', 'Admin'))
    if mode_ == 'Admin':
        st.title('Admin')
        option = st.selectbox(
            '', ('View Inventory',
                 'Update Product',
                 'Delete Product',
                 'Create User',
                 'Generate Report',
                 'Delete User'))
        st.title(option)
        if option == 'View Inventory':
            ALL_DATA = view_all()
            # for i in ALL_DATA[::-1]:
            #     st.markdown(head_message_temp.format(i[0], i[1], i[2]), unsafe_allow_html=True)
            one = [i[0] for i in ALL_DATA[::-1]]
            two = [i[1] for i in ALL_DATA[::-1]]
            three = [i[2] for i in ALL_DATA[::-1]]
            DATA = {
                'time': three,
                'name': one,
                'count': two
            }
            data_ = pd.DataFrame.from_dict(DATA)
            st.table(data_)
            stat = data_.groupby('name')['count'].sum()
            st.table(pd.DataFrame(stat))

        elif option == "Create User":
            blog_author = st.text_input("Enter User Name", max_chars=50)
            password = st.text_input("Enter Password", max_chars=50, type='password')
            if st.button('Create') and blog_author and password:
                st.success('User Created')
        elif option == "Generate Report":
            if st.button('Generate'):
                st.write('Report')
        elif option == 'Delete User':
            name_23 = st.text_input("User Name", max_chars=50)
            if st.button('Delete') and name_23:
                st.success('user deleted')
        elif option == "Update Product":
            name = st.text_input("Product name", max_chars=50)
            if st.button('Update') and name:
                st.success('Product Updated')
        elif option == 'Delete Product':
            name_ = st.text_input("Product name", max_chars=50)
            if st.button('Update') and name_:
                st.success('Product Deleted')
    elif mode_ == 'Staff':
        date = st.sidebar.date_input("Date")
        st.title('Staff')
        IMAGE_SIZE = 640  # default image size
        model = __model()  # model instance
        # change here for confidence of object predict in image
        # by default its 70
        CONF_THR = 0.65  # Confidence threshold

        confidence_threshold = st.sidebar.slider(
            "Confidence threshold", 0.0, 1.0, CONF_THR, 0.05
        )  # Slide bar
        # mode = st.sidebar.radio(
        #     "View Mode", ('🎥 video', '🖼️ image', '📊 data'))
        mode = st.sidebar.radio(
            "View Mode", ('🎥 video', '📊 data', '🖼️ image'))
        if mode == '🎥 video':
            button_placeholder = st.empty()
            st.title("🎥 Object detection video")
            webrtc_ctx = webrtc_streamer(
                key="object-detection",
                mode=WebRtcMode.SENDRECV,
                client_settings=WEBRTC_CLIENT_SETTINGS,
                video_processor_factory=VideoProcessor,
                async_processing=True,
            )
            # if st.button('Capture'):
            #     Flag += 1
            if webrtc_ctx.video_processor:
                # checks if camera is running
                webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
            if st.checkbox("Store", value=False):
                Flag = 1
            # if st.checkbox("Stop it", value=False):
            #     Flag = 0
            if st.checkbox("Show the detected labels", value=True):
                if webrtc_ctx.state.playing:
                    labels_placeholder = st.empty()
                    # button_placeholder = st.empty()
                    empty = st.empty()
                    while True:
                        if webrtc_ctx.video_processor:
                            # webrtc_ctx.video_processor.type = st.radio(
                            #     "Capture", ("No", "Yes")
                            #     )
                            try:
                                result = webrtc_ctx.video_processor.result_queue.get(
                                    timeout=1.0
                                )
                            except queue.Empty:
                                result = None
                            if result:
                                data_ = pd.DataFrame(result[0], index=['items'])
                                labels_placeholder.table(data_)
                                if Flag:
                                    for name, d in result[0].items():
                                        add_data(name, d, date)
                                    Flag = 0
                                    # Flag -= 1
                            else:
                                labels_placeholder.table(result)
                        else:
                            break
            # if st.button('Store'):
            #     Flag += 1
        elif mode == '🖼️ image':
            st.title("🖼️ Object detection image")
            img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if img_file_buffer is not None:
                image = np.array(PIL.Image.open(img_file_buffer))  # Open buffer
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # resize image
                image_box, counting = load(model, image, confidence_threshold,
                                           IMAGE_SIZE)  # function to predict on image
                st.image(
                    image_box, caption=f"Processed image", use_column_width=True,
                )
                C = {k: v for k, v in counting.items() if v > 0}
                data = pd.DataFrame(C, index=['items'])
                st.sidebar.table(data)
                for name, d in C.items():
                    add_data(name, d, date)
        elif mode == '📊 data':
            st.title("📊 data")
            ALL_DATA = view_all()
            # for i in ALL_DATA[::-1]:
            #     st.markdown(head_message_temp.format(i[0], i[1], i[2]), unsafe_allow_html=True)
            one = [i[0] for i in ALL_DATA[::-1]]
            two = [i[1] for i in ALL_DATA[::-1]]
            three = [i[2] for i in ALL_DATA[::-1]]
            DATA = {
                'time': three,
                'name': one,
                'count': two
            }
            data_ = pd.DataFrame.from_dict(DATA)
            st.table(data_)
            stat = data_.groupby('name')['count'].sum()
            st.table(pd.DataFrame(stat))
            Downloadmode = st.sidebar.radio(
            "Download Mode", ('None','Excel', 'CSV'))
            if Downloadmode=='CSV':
                 csvformat(data_)
                 st.write('Data is written successfully to csv File.')

            elif Downloadmode=='Excel':
                excelformat(data_)
                st.write('Data is written successfully to Excel File.')
