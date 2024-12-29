# streamlit_app.py

import streamlit as st
import threading
import asyncio
import cv2
import time

from main import AudioLoop

if "agent_thread" not in st.session_state:
    st.session_state.agent_thread = None
if "stop_agent" not in st.session_state:
    st.session_state.stop_agent = False

st.title("Démonstration - Agent Google + Audio/Video temps réel")

placeholder_camera = st.empty()  

def run_agent():
    """
    Fonction lancée dans un thread pour exécuter l'AudioLoop (caméra + micro + Gemini).
    """
    loop = AudioLoop(video_mode="camera")

    asyncio.run(loop.run())  

def start_conversation():
    """
    Démarre le thread si pas déjà démarré.
    """
    if st.session_state.agent_thread is None or not st.session_state.agent_thread.is_alive():
        st.session_state.stop_agent = False
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        st.session_state.agent_thread = thread
        st.info("Agent démarré (caméra + audio). Regardez la console pour la sortie texte.")

def stop_conversation():
    """
    Demande l'arrêt de l'agent.
    """
    st.session_state.stop_agent = True
    st.warning("Arrêt demandé (mais il faut peut-être fermer la console manuellement)")

col1, col2 = st.columns(2)
with col1:
    if st.button("Démarrer la conversation"):
        start_conversation()

with col2:
    if st.button("Arrêter la conversation"):
        stop_conversation()
