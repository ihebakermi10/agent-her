
import argparse
import asyncio
from config import DEFAULT_MODE
from audio import AudioLoop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=DEFAULT_MODE, type=str)
    args = parser.parse_args()
    main_loop = AudioLoop(video_mode=args.mode)
    asyncio.run(main_loop.run())
