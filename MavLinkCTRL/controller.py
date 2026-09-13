import math
import threading
import time

from pymavlink import mavutil


class DroneController:
    """MAVLink drone controller for ArduPilot SITL / real vehicles."""

    def __init__(self, connection_string: str = 'udpin:localhost:14550'):
        self.telemetry = {}
        self.telemetry_lock = threading.Lock()
        print(f"Connecting to drone on {connection_string}...")
        self.drone = mavutil.mavlink_connection(connection_string)
        self.drone.wait_heartbeat()
        print(f"Connected: system {self.drone.target_system}")

    def __enter__(self):
        self.wait_gps_fix()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.drone.close()
        print("Connection closed.")

    def _wait_mode(self, mode_name: str) -> None:
        """
        Wait for the drone to switch to the specified mode.

        Args:
            mode_name: The name of the mode to wait for (e.g., 'GUIDED', 'RTL').
        Returns:
            None
        """
        print(f"Waiting for mode {mode_name}...")
        target_mode_id = self.drone.mode_mapping()[mode_name]
        while True:
            msg = self.drone.recv_match(type='HEARTBEAT', blocking=True)
            if msg.custom_mode == target_mode_id:
                print(f"Mode {mode_name} confirmed!")
                break

    def _wait_altitude(
            self,
            target_altitude: float,
            epsilon: float = 0.3
    ) -> None:
        """
        Wait until the drone reaches the target altitude (relative to takeoff point) within a certain tolerance.

        Args:
            target_altitude: The target altitude in meters.
            epsilon: The acceptable deviation from the target altitude in meters (default is 0.3m).
        Returns:
            None
        """
        print("Reaching target altitude...")
        while True:
            msg = self.drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            # relative_alt in millimeters, convert to meters
            current_alt = msg.relative_alt / 1000.0
            if abs(current_alt - target_altitude) <= epsilon:
                print(f"Target altitude reached! Current altitude: {current_alt:.2f}m")
                break

    def _wait_location(
            self,
            target_lat: float,
            target_lon: float,
            epsilon_meters: float = 1.5,
            blocking: bool = True
    ) -> bool:
        """
        Wait until the drone reaches the target GPS coordinates within a certain distance (in meters).

        Args:
            target_lat: Target latitude in decimal degrees.
            target_lon: Target longitude in decimal degrees.
            epsilon_meters: Acceptable distance to target in meters (default is 1.5m).
        Returns:
            None
        """
        while True:
            cur_lat, cur_lon = self.check_location()

            # Simple distance calculation (not accounting for Earth's curvature, but sufficient for small distances)
            dist = math.sqrt((target_lat - cur_lat) ** 2 + (target_lon - cur_lon) ** 2) * 111319.5
            if dist <= epsilon_meters:
                print(f"Target location reached! Distance: {dist:.1f}m")
                break
            if not blocking:
                print(f"Current distance to target: {dist:.1f}m", end='\r')
                break
        return dist <= epsilon_meters

    def get_battery_status(self) -> int | None:
        """
        Get the current battery status.

        Returns:
            Battery level as a percentage (0-100) if available, otherwise None.
        """
        level = None
        msg = self.drone.recv_match(type='SYS_STATUS', blocking=True)
        if msg is not None:
            voltage = msg.voltage_battery / 1000.0
            current = msg.current_battery / 100.0
            level = msg.battery_remaining
            print(f"Battery status: {level}% (Voltage: {voltage:.2f}V, Current: {current:.2f}A)")
        else:
            print("Failed to get battery status.")
        return level if msg is not None else 0

    def get_telemetry(self) -> dict:
        """
        Get the current telemetry data including location, altitude, and battery status.

        Returns:
            A dictionary containing 'latitude', 'longitude', 'altitude', and 'battery' keys.
        """
        self.telemetry = {}
        msg = self.drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg is not None:
            self.telemetry['latitude'] = msg.lat / 1e7
            self.telemetry['longitude'] = msg.lon / 1e7
            self.telemetry['altitude'] = msg.relative_alt / 1000.0  # in meters
        else:
            print("Failed to get location data.")
            self.telemetry['latitude'] = 0.0
            self.telemetry['longitude'] = 0.0
            self.telemetry['altitude'] = 0.0

        battery_level = self.get_battery_status()
        self.telemetry['battery'] = battery_level if battery_level is not None else 0

        return self.telemetry


    def check_location(self) -> tuple:
        """
        Check the current GPS location of the drone and print it.

        Returns:
            A tuple of (latitude, longitude) in decimal degrees and meters.
        """
        telemetry = self.get_telemetry()
        return telemetry['latitude'], telemetry['longitude']

    def wait_gps_fix(self):
        """
        Wait for GPS fix and EKF initialization to ensure the drone is ready for flight.

        Returns:
            None
        """
        print("Waiting for GPS fix...")
        while True:
            msg = self.drone.recv_match(type='GPS_RAW_INT', blocking=True)
            if msg.fix_type >= 3:
                print(f"GPS Fix received: {msg.fix_type}D")
                break

        # Additional check for EKF status to ensure the drone's state estimation is reliable
        while True:
            msg = self.drone.recv_match(type='EKF_STATUS_REPORT', blocking=True)
            # Check if the EKF flags indicate a good state estimation (flags >= 448 means EKF is ready for flight)
            if msg.flags >= 448:
                print("EKF ready for flight.")
                break

    # --- Main drone control methods ---

    def set_mode(self, mode_name: str) -> None:
        """
        Set the flight mode of the drone.

        Args:
            mode_name: The name of the mode to set (e.g., 'GUIDED', 'RTL').
        Returns:
            None
        """
        if mode_name not in self.drone.mode_mapping():
            print(f"Unknown mode: {mode_name}")
            return
        mode_id = self.drone.mode_mapping()[mode_name]
        self.drone.mav.set_mode_send(
            self.drone.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        self._wait_mode(mode_name)

    def arm(self):
        """
        Arm the drone to start the motors.

        Returns:
            None
        """
        print("Arming...")
        self.drone.mav.command_long_send(
            self.drone.target_system, self.drone.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        self.drone.motors_armed_wait()
        print("Motors started.")

    def takeoff(self, target_altitude: float) -> None:
        """
        Takeoff to a specified altitude (relative to takeoff point).

        Args:
            target_altitude:

        Returns:
            None
        """
        print(f"Take off to {target_altitude} meters...")
        self.drone.mav.command_long_send(
            self.drone.target_system, self.drone.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, target_altitude
        )
        # Waiting for the takeoff command to be acknowledged
        ack = self.drone.recv_match(type='COMMAND_ACK', blocking=True)
        print(f"ACK received for takeoff command: {ack.result}")
        self._wait_altitude(target_altitude)
        print(f"Takeoff complete. Current altitude: {target_altitude}m")

    def arm_and_takeoff(self, altitude: int = 3):
        """
        Arm the drone and take off to a specified altitude.

        Args:
            altitude: Target altitude in meters (relative to takeoff point).
        Returns:
            None
        """
        self.set_mode('GUIDED')
        self.arm()
        # self.takeoff(altitude)

    def disarm(self) -> None:
        """
        Drone disarming to stop the motors.

        Returns:
            None
        """
        print("Disarming...")
        self.drone.mav.command_long_send(
            self.drone.target_system, self.drone.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self.drone.motors_disarmed_wait()
        print("Motors stopped.")

    def send_drone_by_coords(
            self,
            lat: float,
            lon: float,
            alt: float,
            blocking: bool = True
    ) -> None:
        """
        Fly to a specific GPS coordinate (latitude, longitude) at a given altitude (relative to takeoff point).

        Args:
            lat: Target latitude in decimal degrees.
            lon: Target longitude in decimal degrees.
            alt: Target altitude in meters (relative to takeoff point).
            blocking: If True, wait until the drone reaches the target location before returning. If False, return immediately after sending the command.

        Returns:
            None
        """
        self.drone.mav.set_position_target_global_int_send(
            0, self.drone.target_system, self.drone.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000,
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        print(f"Target coordinates sent: {lat}, {lon} at {alt}m altitude.")
        if blocking:
            self._wait_location(lat, lon)
            print("Arrived at target coordinates.")

    # def set_gimbal_angle(self, pitch, roll=0, yaw=0):
    #     # pitch у градусах (наприклад, -45)
    #     self.drone.mav.command_long_send(
    #         self.drone.target_system, self.drone.target_component,
    #         mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
    #         0,
    #         pitch * 100, roll * 100, yaw * 100,  # Команди йдуть у центиградусах (де 100 = 1 градус)
    #         0, 0, 0,
    #         mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING
    #     )

    def send_drone_velocity(
            self,
            vx: float,
            vy: float,
            vz: float = 0.0,
    ) -> None:
        """
        Send velocity commands to the drone in the local NED frame for a specified duration.

        Args:
            vx: Velocity in the North direction (m/s).
            vy: Velocity in the East direction (m/s).
            vz: Velocity in the Down direction (m/s) (positive down).
        Returns:
            None
        """
        self.drone.mav.set_position_target_local_ned_send(
            0,
            self.drone.target_system,
            self.drone.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,  # маска — тільки velocity
            0, 0, 0,  # position (ігнорується)
            vx, vy, vz,  # velocity
            0, 0, 0,  # acceleration (ігнорується)
            0, 0  # yaw, yaw_rate
        )

    def hover(self) -> None:
        """
        Command the drone to hover in place by sending zero velocity.

        Returns:
            None
        """
        self.send_drone_velocity(0, 0, 0)

    def smart_patrol(
            self,
            waypoints,
            min_battery: int = 30
    ) -> None:
        """
        Patrol through a list of waypoints while continuously monitoring battery status.
        If the battery level drops below the specified threshold, the drone will return home (RTL).

        Args:
            waypoints: A list of tuples, where each tuple contains (latitude, longitude, altitude) for the patrol points.
            min_battery: Minimum battery percentage required to continue patrolling.
            If the battery level drops below this threshold, the drone will return home (default is 30%).
        Returns:
            None
        """
        print(f"Patrolling through {len(waypoints)} waypoints with minimum battery threshold of {min_battery}%...")

        while True:
            for i, wp in enumerate(waypoints):
                lat, lon, alt = wp
                print(f"Moving to waypoint {i + 1}/{len(waypoints)}: {lat}, {lon}, {alt}m")

                # Send the drone to the next waypoint without blocking, so we can monitor battery status during flight
                drone.send_drone_by_coords(lat, lon, alt, blocking=False)

                # Waiting for the drone to reach the waypoint while monitoring battery status
                while True:
                    # 1. Battery check
                    batt = self.get_battery_status()
                    if batt is not None:
                        if batt < min_battery:
                            print(f"!!! CRITICAL: Battery low ({batt}%). Returning home (RTL) !!!")
                            drone.rtl()  # Send command to return home
                            return
                        print(f"STATUS: Battery at {batt}%")

                    # 2. Checking if the drone has reached the waypoint (non-blocking)
                    is_done = drone._wait_location(lat, lon, epsilon_meters=2.0, blocking=False)
                    if is_done:
                        break
                    time.sleep(1)  # Waiting a bit before the next check

    def rtl(self) -> None:
        """
        Return to Launch (RTL) mode to safely return the drone home.

        Returns:
            None
        """
        print("Returning to Launch (RTL)...")
        self.set_mode('RTL')
        print("Waiting for the drone to return home...")
        self.drone.motors_disarmed_wait()
        print("Drone has returned home and disarmed.")


# --- EXAMPLE ---

if __name__ == "__main__":
    with DroneController() as drone:
        drone.arm_and_takeoff(altitude=3)

        drone.patrol_points = [
            (-35.359662, 149.153259, 15),
            (-35.353887, 149.162783, 15),
        ]
        drone.smart_patrol(drone.patrol_points, min_battery=50)
