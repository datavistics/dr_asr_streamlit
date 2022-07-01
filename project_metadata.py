import streamlit as st  # willl be redundant in main program

DEPLOYMENT_ID = st.secrets["DEPLOYMENT_ID"]
API_URL = f'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{DEPLOYMENT_ID}/predictionsUnstructured'    # noqa
API_KEY = st.secrets["API_KEY"]
DATAROBOT_KEY = st.secrets["DATAROBOT_KEY"]
