
import asyncio
import pyaudio
from config import FORMAT, CHANNELS, SEND_SAMPLE_RATE, CHUNK_SIZE

async def listen_audio(out_queue: asyncio.Queue):
    pya_ = pyaudio.PyAudio()
    mic_info = pya_.get_default_input_device_info()
    audio_stream = await asyncio.to_thread(
        pya_.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )
    kwargs = {"exception_on_overflow": False}
    try:
        while True:
            data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
            await out_queue.put({"data": data, "mime_type": "audio/pcm"})
    except asyncio.CancelledError:
        audio_stream.close()
