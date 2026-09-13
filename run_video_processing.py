import threading

from gi.repository import GLib

from Tracker import CameraTracker

with CameraTracker(host='127.0.0.1', port=5000) as tracker:
    yolo_thread = threading.Thread(target=tracker.yolo_worker, daemon=True)
    yolo_thread.start()
    loop = GLib.MainLoop()
    loop.run()
