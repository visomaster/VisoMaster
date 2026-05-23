"""Run the streamrelay server and a local OpenCV viewer in one process.

Useful as a smoke test:

    python examples/standalone_viewer.py

then point your phone's browser at https://<this-host>:9090/
"""

from __future__ import annotations

import multiprocessing as mp
import time

import cv2

from streamrelay import FrameReader, StreamServer

SHM_NAME = "streamrelay_demo"


def _serve():
    StreamServer(shm_name=SHM_NAME).run()


def main():
    server = mp.Process(target=_serve, daemon=True)
    server.start()

    try:
        reader = FrameReader(shm_name=SHM_NAME, attach_timeout=15.0)
    except FileNotFoundError as e:
        print(e)
        return

    print("Server up. Open https://<this-host>:9090/ on your phone.")
    while True:
        frame = reader.read_new()
        if frame is None:
            time.sleep(0.005)
            continue
        cv2.imshow("streamrelay", frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    reader.close()


if __name__ == "__main__":
    main()
