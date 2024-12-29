
import asyncio
import cv2
import PIL.Image
import io
import base64

async def get_frames(out_queue: asyncio.Queue):
    cap = await asyncio.to_thread(cv2.VideoCapture, 0)
    try:
        while True:
            frame = await asyncio.to_thread(_get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await out_queue.put(frame)
    except asyncio.CancelledError:
        pass
    finally:
        cap.release()

def _get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(frame_rgb)
    img.thumbnail([1024, 1024])
    image_io = io.BytesIO()
    img.save(image_io, format="jpeg")
    image_io.seek(0)
    mime_type = "image/jpeg"
    image_bytes = image_io.read()
    return {
        "mime_type": mime_type,
        "data": base64.b64encode(image_bytes).decode()
    }
