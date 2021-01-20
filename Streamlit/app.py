import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io
import os
import base64
import predict_function
from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, MODEL, PROTOTXT
from minio import Minio
import pandas as pd
import time
from object_detection.utils import label_map_util


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def get_image_download_link(img):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}"><input type="button" value="Download"></a>'
    return href
    

@st.cache(allow_output_mutation=True)
def get_cap(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))

    # Check if camera opened successfully
    if (video_stream.isOpened() == False):
        print("Error opening video  file")
    return video_stream

@st.cache
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


@st.cache
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
    return image, labels


st.title("口罩偵測")
st.write("""
 這個app可偵測有沒有戴口罩\n
 支援照片和影片檢測
""")


DEMO_IMAGE='sampleImage.png'
PATH_TO_LABELS="model/annotations/label_map.pbtxt"
services=['image','video_real_time','video_off_line']
option = st.selectbox('',services)

model=predict_function.openModel()
category_index=label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

if(option=='image'):
    #st.write("上傳照片:")
    # start predict
    img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image_orignal=Image.open(img_file_buffer)
        image=np.array(Image.open(img_file_buffer).convert("RGB"))
        image_res=predict_function.detectImage(image,model,category_index)
        st.image(
        image_res, caption=f"Upload image", use_column_width=True,
        )
    else:
        demo_image = DEMO_IMAGE
        image_orignal=Image.open(demo_image)
        image=np.array(Image.open(demo_image).convert("RGB"))
        image_res=predict_function.detectImage(image,model,category_index)
        st.image(
        image_res, caption=f"Sample image", use_column_width=True,
        )
    result = Image.fromarray(image_res)
    #st.markdown(get_image_download_link(result), unsafe_allow_html=True)

elif(option=='video_real_time'):

    #pause
    #stop
    video_file_buffer = st.file_uploader("", type=["mp4", "avi"])
    temporary_location=False
    if video_file_buffer is not None:
        g = io.BytesIO(video_file_buffer.read())  ## BytesIO Object
        temporary_location = "testout_simple.mp4"
        
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
        # close file
        out.close()
    
    scaling_factorx = 0.25
    scaling_factory = 0.25
    image_placeholder = st.empty()
    
    
    if temporary_location:
        frame_index=0
        frame_index_stat = st.empty()
        exe_time_stat = st.empty()
        fps_stat = st.empty()
        while True:
            # here it is a CV2 object
            video_stream = get_cap(temporary_location)
            # video_stream = video_stream.read()
            ret, image = video_stream.read()
            frame_index+=1
            before_time=time.time()
            
            if ret:
                image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)
                image_res=predict_function.detectVideo_online(image,model,category_index)
            else:
                print("there was a problem or video was finished")
                cv2.destroyAllWindows()
                video_stream.release()
                break
            # check if frame is None
            if image is None:
                print("there was a problem None")
                # if True break the infinite loop
                break
            image_placeholder.image(image_res, channels="BGR", use_column_width=True)
            after_time=time.time()
            exe_time=round((after_time-before_time) * 1000,2)
            frame_index_stat.text('Frame: %s' % frame_index)
            exe_time_stat.text('time:'+str(exe_time)+" ms")
            fps_stat.text('fps:'+str(round(1000/exe_time,1)))
            
            
            
            cv2.destroyAllWindows()
        video_stream.release()
        cv2.destroyAllWindows()
    
elif(option=='video_off_line'):
    video_file_buffer = st.file_uploader("", type=["mp4", "avi"])
    temporary_location=None
    if video_file_buffer is not None:
        g = io.BytesIO(video_file_buffer.read())  ## BytesIO Object
        temporary_location = "test_video.mp4"
        
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
        # close file
        out.close()
    
    #scaling_factorx = 0.25
    #scaling_factory = 0.25
    image_placeholder = st.empty()
    
    #warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
    input_video_name=temporary_location
    cap = cv2.VideoCapture(input_video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter('Temp.mp4',fourcc, fps, size)

    
    frame_number=0 
    if temporary_location:
        step1_stat = st.empty()
        step1_stat.text("""Step 1: 進行物件偵測...\n""")
        my_bar = st.progress(0)
        step2_stat = st.empty()
        while cap.isOpened():
            my_bar.progress(frame_number/length)
            frame_number+=1
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            image_res=predict_function.detectVideo_online(frame,model,category_index)
            res = cv2.resize(image_res, size, interpolation=cv2.INTER_CUBIC)
    
            #print("Writing frame {} / {}".format(frame_number, length))
            out.write(res)

        out.release()
        step2_stat.text("""Step 2: 將影片轉成html播放器支援的格式...\n""")
        os.system("ffmpeg -y -i Temp.mp4 -vcodec libx264 Temp2.mp4")
        step2_stat.text("""Step 2: 轉檔完成\n""")
        st.video('Temp2.mp4')
    pass




