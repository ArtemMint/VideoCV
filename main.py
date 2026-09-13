import threading
import time

from gi.repository import GLib

from MavLinkCTRL import DroneController
from Tracker import CameraTracker
from Tracker.controller import PID

FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2
DEAD_ZONE = 10
TARGET_AREA = 0.8

pid_x = PID(kp=0.005, ki=0.0001, kd=0.001, limit=1.0)
pid_y = PID(kp=0.005, ki=0.0001, kd=0.001, limit=1.0)
pid_z = PID(kp=2, ki=0.0001, kd=0.5, limit=3.0)  #


def control_loop(drone: DroneController, tracker: CameraTracker):
    while True:
        target = next((t for t in tracker.latest_tracks if t[4] == 'truck'), None)  # find first target

        if target:
            x1, y1, x2, y2, cls, conf, tid = target
            target_x = (x1 + x2) // 2
            target_y = (y1 + y2) // 2

            error_x = target_x - CENTER_X
            error_y = target_y - CENTER_Y

            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = FRAME_W * FRAME_H
            relative_size = bbox_area / frame_area
            error_z = TARGET_AREA - relative_size

            print(f"Target #{tid} ({cls}): center=({target_x},{target_y}) error=({error_x},{error_y})")

            vx, vy, vz = 0.0, 0.0, 0.0

            # бічне зміщення → vy
            if abs(error_x) > DEAD_ZONE:
                vy = pid_x.compute(error_x, dt=0.1)

            # вертикальне зміщення → vz (NED: негативне = вгору)
            if abs(error_y) > DEAD_ZONE:
                vz = pid_y.compute(error_y, dt=0.1)

            # рух до цілі → vx, тільки коли ціль в центрі
            if abs(error_x) < DEAD_ZONE and abs(error_y) < DEAD_ZONE:
                if abs(error_z) > 0.02:
                    vx = pid_z.compute(error_z, dt=0.1)

            drone.send_drone_velocity(vx, vy, vz)
        else:
            print("No target")
            pid_x.reset()
            pid_y.reset()
            pid_z.reset()
            drone.hover()

        time.sleep(0.1)


def telemetry_loop(drone: DroneController, tracker: CameraTracker):

    while True:
        try:
            telemetry = drone.get_telemetry()
            tracker.update_telemetry(telemetry)
        except Exception as e:
            print(f"Telemetry error: {e}")
        time.sleep(0.2)


with DroneController() as drone:
    drone.arm_and_takeoff(altitude=1)

    with CameraTracker(host='127.0.0.1', port=5000) as tracker:
        # Stream processing in a separate thread and push drone telemetry updates in the main thread
        yolo_thread = threading.Thread(target=tracker.yolo_worker, daemon=True)
        yolo_thread.start()

        ctrl_thread = threading.Thread(target=control_loop, args=(drone, tracker), daemon=True)
        ctrl_thread.start()

        telemetry_thread = threading.Thread(target=telemetry_loop, args=(drone, tracker), daemon=True)
        telemetry_thread.start()

        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            print("Stopping...")
            drone.hover()
            drone.rtl()
