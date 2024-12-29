
import asyncio
from config import DEFAULT_MODE, MODEL, CONFIG
from audio.audio_handling import listen_audio
from audio.audio_player import play_audio
from video.video_handling import get_frames
from tools.handle_tool import handle_tool_call
from google import genai
import traceback

from tool import tweet_posting_tool

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.strip().lower() == "q":
                break
            await self.session.send(text or ".", end_of_turn=True)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(msg)

    async def receive_audio(self):
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                if text := response.text:
                    print(text, end="")

                if response.tool_call:
                    await handle_tool_call(self.session, response.tool_call)

            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def run(self):
        try:
            client = genai.Client(http_options={"api_version": "v1alpha"})
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                init_msg = "Bonjour! presente vous '."
                await self.session.send(init_msg, end_of_turn=True)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(listen_audio(self.out_queue))

                if self.video_mode == "camera":
                    tg.create_task(get_frames(self.out_queue))
                elif self.video_mode == "screen":
                    tg.create_task(get_screen_async(self.out_queue))

                tg.create_task(self.receive_audio())
                tg.create_task(play_audio(self.audio_in_queue))

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except genai.ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)
