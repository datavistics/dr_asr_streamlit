from datetime import datetime
import logging
import logging.handlers
import queue
import threading
import time
import numpy_serializer as ns
from collections import deque
from pathlib import Path
from pprint import pformat
from typing import List
import json

import av
import numpy as np
import pydub
import requests
import streamlit as st

from streamlit_webrtc import WebRtcMode, webrtc_streamer

from project_metadata import (
    API_URL,
    API_KEY,
    DEPLOYMENT_ID,
    DATAROBOT_KEY,
)

HERE = Path(__file__).parent
SAMPLE_RATE = 16000
AUDIO_RECEIVER_SIZE = 2048


#################
# Change these
#################
TOTAL_LINES = 10
TIME_TO_COLLECT_AUDIO = 2.0

logger = logging.getLogger(__name__)


proj_dir = Path(__file__).parent
proj_dir

server_request = requests.get('https://raw.githubusercontent.com/pradt2/always-online-stun/master/valid_hosts.txt')
server_list = ["stun:stun.l.google.com:19302"] + server_request.text.split('\n')

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501

def main():
    st.header("Real Time Speech-to-Text")
    st.markdown(
        """
This demo app is using [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h) from Facebook,
an open speech-to-text engine.

![wav2vec2](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/wav2vec2.png)

I deployed it with DataRobot as an Unstructured [Custom Inference Model](https://docs.datarobot.com/en/docs/mlops/deployment/custom-models/custom-inf-model.html), 
and wrote a streamlit app hosted by DataRobot's AI Code Apps. 

How to use:
1. Select the app mode (`Sound only` works a bit better)
2. Configure the settings to your liking.
3. Wait for the status to say `Running. Say something!`
    - If you get an error `No frame arrived.` you need to choose a different server
4. Start speaking!
"""
    )

    sound_only_page = "Sound only"
    with_video_page = "With video"
    app_mode = st.selectbox("Choose the app mode", [sound_only_page, with_video_page])

    total_lines = st.slider("Total Lines printed", 1, 20, TOTAL_LINES, step=1)
    time_to_collect_audio = st.slider("Time to collect audio frames", 0.2, 5.0, TIME_TO_COLLECT_AUDIO, step = 0.1)
    server = st.selectbox("STUN Server: (change if you get No Frame Arrived)", server_list)


    if app_mode == sound_only_page:
        app_sst(total_lines, time_to_collect_audio, server)
    elif app_mode == with_video_page:
        app_sst_with_video()

def app_sst(total_lines, time_to_collect_audio, server):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=AUDIO_RECEIVER_SIZE,
        rtc_configuration={"iceServers": [{"urls": [server]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    text_list = []

    while True:
        if webrtc_ctx.audio_receiver:
            time.sleep(time_to_collect_audio)
            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound
            
            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(SAMPLE_RATE)
                buffer = np.array(sound_chunk.get_array_of_samples()).astype(np.double)

                out = {
                    'buffer': ns.to_bytes(buffer).decode('latin-1'),
                    'args': {}
                              }
                
                current_time = datetime.now().strftime("%H:%M:%S")

                data = json.dumps(out)
                text = simple_predict(data)
                text = json.loads(text.decode())
                if 'prediction' not in text:
                    logger.warning(pformat(text))
                    continue
                text = text['prediction']
                if text['text']:
                    text_list = [f"**Text {current_time}:** {text['text']}"] + text_list[:total_lines - 1]
                    text_output.markdown('\n\n'.join(reversed(text_list)))
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


def app_sst_with_video():
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    text_list = []

    while True:
        if webrtc_ctx.state.playing:
            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 200:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(SAMPLE_RATE)
                buffer = np.array(sound_chunk.get_array_of_samples()).astype(np.double)

                out = {
                    'buffer': ns.to_bytes(buffer).decode('latin-1'),
                    'args': {}
                              }
                
                current_time = datetime.now().strftime("%H:%M:%S")

                data = json.dumps(out)
                text = simple_predict(data)
                text = json.loads(text.decode())
                if 'prediction' not in text:
                    logger.warning(pformat(text))
                    continue
                text = text['prediction']
                if text['text']:
                    text_list = [f"**Text {current_time}:** {text['text']}"] + text_list[:4]
                    text_output.markdown('\n\n'.join(reversed(text_list)))
        else:
            status_indicator.write("Stopped.")
            break

        
def simple_predict(data):
    """
    Make unstructured predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://app.datarobot.com/docs/predictions/api/dr-predapi.html
 
    Parameters
    ----------
    data : bytes
        Bytes data read from provided file.
    deployment_id : str
        The ID of the deployment to make predictions with.
    mimetype : str
        Mimetype describing data being sent.
        If mimetype starts with 'text/' or equal to 'application/json',
        data will be decoded with provided or default(UTF-8) charset
        and passed into the 'score_unstructured' hook implemented in custom.py provided with the model.
 
        In case of other mimetype values data is treated as binary and passed without decoding.
    charset : str
        Charset should match the contents of the file, if file is text.
 
    Returns
    -------
    data : bytes
        Arbitrary data returned by unstructured model.
 
 
    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    mimetype = 'text/plain'
    charset = 'utf8'
    headers = {
        'Content-Type': '{};charset={}'.format(mimetype, charset),
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
 
    url = API_URL.format(deployment_id=DEPLOYMENT_ID)
 
    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=data,
        headers=headers,
    )
    # Return raw response content
    return predictions_response.content


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()