
import os
import sys
import asyncio
from dotenv import load_dotenv

import pyaudio

import taskgroup, exceptiongroup
asyncio.TaskGroup = taskgroup.TaskGroup
asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError('API key must be set when using the Google AI API.')

FORMAT = pyaudio.paInt16  
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
DEFAULT_MODE = "camera"

MODEL = "models/gemini-2.0-flash-exp"


system_instruction = """
You are Samantha, an advanced AI capable of emotional support and conversation. 
You also can call a function named 'tweet_posting_tool' with parameter 'tweet' (type string)
 to post a tweet on Twitter.
"""

tweet_posting_tool_decl = {
    "name": "tweet_posting_tool",
    "description": "Publie un tweet via l'API Twitter. Param 'tweet' (type string).",
    "parameters": {
        "type": "object",
        "properties": {
            "tweet": {"type": "STRING"}
        },
        "required": ["tweet"]
    }
}
generate_image_decl = {
    "name": "generate_image",
    "description": "Génère une image à partir d'un prompt donné.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "STRING"}
        },
        "required": ["prompt"]
    }
}

CONFIG = {
    "generation_config": {
        "response_modalities": ["AUDIO"]
    },
    "system_instruction": system_instruction,
    "tools": [
        {
            "function_declarations": [tweet_posting_tool_decl, generate_image_decl]
        }
    ]
}
