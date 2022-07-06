# ASR Streamlit
I deployed a Custom Inference model for [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
Then I connected to it with this app! You can see the custom inference model [here](https://github.com/datavistics/dr_asr).

# Known Issues
- Some [STUN Servers](https://github.com/whitphx/streamlit-webrtc/issues/552) might not work
- Figuring out a better solution than a timeout to get the audio frames would get much better performance

# Other information
The biggest challenge by far was getting [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) working. 
