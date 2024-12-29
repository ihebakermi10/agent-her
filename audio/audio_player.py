
import asyncio
import pyaudio
from config import RECEIVE_SAMPLE_RATE

async def play_audio(audio_in_queue: asyncio.Queue):
    pya_ = pyaudio.PyAudio()
    stream = await asyncio.to_thread(
        pya_.open,
        format=pyaudio.paInt16,
        channels=1,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )
    try:
        while True:
            bytestream = await audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)
    except asyncio.CancelledError:
        stream.close()
