#!/usr/bin/env python3

import sys
import math
import subprocess

from PyQt5 import QtWidgets, QtCore, QtGui

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

# Pfad zu deinem Logo (anpassen falls nötig)
LOGO_PATH = "/home/duckie6/GalapagosGang_URRMC2025/src/tb3_navigation/resource/gg.jpg"


class Nav2GoalClient(Node):
    def __init__(self):
        super().__init__('nav2_goal_client')

        # --- Nav2 Action Client (nur für Idle / manuelles Ziel) ---
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_active = False
        self._goal_handle = None

        # --- Missions ---
        # current_mission = Vorauswahl (per Button)
        # active_mission  = Mission, die tatsächlich gestartet wurde (0 = keine)
        self.current_mission = 0
        self.active_mission = 0
        self.mission_pub = self.create_publisher(Int32, 'mission_id', 10)

        # Optional: Prozesse zu gestarteten Launchfiles
        self._mission_processes = {}

        # Referenz auf GUI (für Status-Updates)
        self.gui = None

    # ------------------------------------------------------------------
    # GUI-Referenz
    # ------------------------------------------------------------------
    def set_gui(self, gui):
        self.gui = gui

    # ------------------------------------------------------------------
    # Mission Handling
    # ------------------------------------------------------------------
    def set_mission(self, mission_id: int):
        """
        Nur die Vorauswahl setzen.
        WICHTIG: Es wird NICHT gepublished – das passiert erst beim Go-Button.
        """
        self.current_mission = mission_id

        if mission_id == 0:
            txt = "Mission selection: Idle (0)"
        else:
            txt = f"Mission selection: {mission_id}"

        self.get_logger().info(txt)

        if self.gui:
            self.gui.set_status(txt)
            self.gui.set_mission_display(mission_id)

    def start_selected_mission(self):
        """
        Wird NUR für Missionen >0 genutzt.
        - ggf. vorher laufende Mission stoppen
        - zugehörige Nodes starten (per Launchfile o.ä.)
        - mission_id publishen, damit control_node loslegen kann
        """
        if self.current_mission == 0:
            # Idle wird separat behandelt (direktes Nav2-Goal)
            self.get_logger().info(
                "start_selected_mission() aufgerufen, aber current_mission=0 (Idle) -> ignoriere."
            )
            return

        # Falls noch eine andere Mission aktiv ist, stoppen wir sie erst
        if self.active_mission not in (0, self.current_mission):
            self.stop_active_mission()

        mission_id = self.current_mission

        # 1) Nodes für diese Mission starten
        self.launch_nodes_for_mission(mission_id)

        # 2) mission_id publishen
        msg = Int32()
        msg.data = mission_id
        self.mission_pub.publish(msg)

        self.active_mission = mission_id

        txt = f"Mission {mission_id} STARTED (mission_id={mission_id})."
        self.get_logger().info(txt)

        if self.gui:
            self.gui.set_status(txt)
            self.gui.set_mission_display(mission_id)

    def stop_active_mission(self):
        """
        Aktive Mission (falls != 0) stoppen.
        - mission_id=0 publishen (Idle)
        - ggf. gestartete Prozesse beenden (falls konfiguriert)
        """
        if self.active_mission == 0:
            self.get_logger().info("Keine aktive Mission zum Stoppen.")
            return

        mission_id = self.active_mission
        self.get_logger().info(f"Stoppe Mission {mission_id} -> mission_id=0 (Idle).")

        # mission_id=0 publishen (Control-Node kann darauf reagieren)
        msg = Int32()
        msg.data = 0
        self.mission_pub.publish(msg)
        self.active_mission = 0

        # Falls wir Prozesse für Missionen gestartet haben, können wir sie hier beenden
        proc = self._mission_processes.get(mission_id)
        if proc is not None:
            self.get_logger().info(f"Beende Launch-Prozess für Mission {mission_id}.")
            try:
                proc.terminate()
            except Exception as e:
                self.get_logger().error(f"Fehler beim Beenden von Mission {mission_id}: {e}")

    def launch_nodes_for_mission(self, mission_id: int):
        """
        Startet die notwendigen Nodes für eine Mission.
        HIER trägst du deine eigenen ros2 launch Befehle ein.
        """
        # Mapping Mission -> Launch-Command (BEISPIELE, bitte anpassen!)
        launch_cmd = None

        if mission_id == 1:
            # TODO: an dein Paket / Launchfile anpassen
            launch_cmd = ["ros2", "launch", "galapagos_missions", "mission1_lane_follow.launch.py"]
        elif mission_id == 2:
            launch_cmd = ["ros2", "launch", "galapagos_missions", "mission2_something.launch.py"]
        elif mission_id == 3:
            launch_cmd = ["ros2", "launch", "galapagos_missions", "mission3_something.launch.py"]
        elif mission_id == 4:
            launch_cmd = ["ros2", "launch", "galapagos_missions", "mission4_something.launch.py"]

        if launch_cmd is None:
            self.get_logger().warn(
                f"Keine Launch-Command für Mission {mission_id} definiert. "
                f"Bitte in launch_nodes_for_mission() anpassen."
            )
            if self.gui:
                self.gui.set_status(f"No launch command defined for mission {mission_id}.")
            return

        self.get_logger().info(f"Starte Nodes für Mission {mission_id}: {' '.join(launch_cmd)}")
        try:
            proc = subprocess.Popen(launch_cmd)
            self._mission_processes[mission_id] = proc
        except Exception as e:
            self.get_logger().error(
                f"Fehler beim Starten der Launch-Command für Mission {mission_id}: {e}"
            )
            if self.gui:
                self.gui.set_status(f"Error launching mission {mission_id}: {e}")

    # ------------------------------------------------------------------
    # Nav2 Goal Handling (nur für Idle)
    # ------------------------------------------------------------------
    def send_goal(self, x: float, y: float, yaw: float):
        """
        Goal an Nav2 schicken (x, y in m, yaw in rad im map-Frame).
        Wird NUR genutzt, wenn current_mission == 0 (Idle).
        """
        if self.current_mission != 0:
            self.get_logger().warn(
                "send_goal() aufgerufen, aber current_mission != 0. Ignoriere Nav2-Goal."
            )
            return

        if self.goal_active:
            self.get_logger().warn("Schon ein Goal aktiv, ignoriere neues Goal.")
            if self.gui:
                self.gui.set_status("Goal already active.")
            return

        # check: ist der Action-Server da?
        if not self._action_client.wait_for_server(timeout_sec=0.1):
            msg = "navigate_to_pose Action-Server nicht verfügbar."
            self.get_logger().error(msg)
            if self.gui:
                self.gui.set_status(msg)
            return

        # Goal-Nachricht bauen
        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        # yaw (rad) -> Quaternion (nur Drehung um Z)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal_msg.pose = pose

        self.goal_active = True
        self._goal_handle = None
        if self.gui:
            self.gui.set_status("Goal gesendet, warte auf Akzeptanz...")

        # Goal asynchron senden
        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Wird aufgerufen, wenn Nav2 das Goal angenommen/abgelehnt hat."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.goal_active = False
            self._goal_handle = None
            self.get_logger().warn("Goal von Nav2 abgelehnt.")
            if self.gui:
                self.gui.set_status("Goal rejected.")
            return

        self.get_logger().info("Goal akzeptiert, Navigation läuft...")
        self._goal_handle = goal_handle
        if self.gui:
            self.gui.set_status("Goal accepted, navigating...")

        # Auf Ergebnis warten
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        """Wird aufgerufen, wenn Nav2 fertig ist (erreicht / fehlgeschlagen / abgebrochen)."""
        self.goal_active = False
        self._goal_handle = None
        try:
            result = future.result().result  # Struktur hängt von Nav2-Version ab
            self.get_logger().info(f"Navigation beendet: {result}")
            if self.gui:
                self.gui.set_status("Arrived at goal.")
        except Exception as e:
            self.get_logger().error(f"Fehler beim Holen des Ergebnisses: {e}")
            if self.gui:
                self.gui.set_status(f"Error: {e}")

    def feedback_callback(self, feedback_msg):
        """Laufendes Feedback von Nav2 (z.B. distance_remaining)."""
        fb = feedback_msg.feedback
        dist = getattr(fb, "distance_remaining", None)
        if dist is not None and self.gui:
            self.gui.set_status(f"Navigating... remaining {dist:.2f} m")

    def cancel_goal(self):
        """Aktives Nav2-Goal abbrechen."""
        if not self.goal_active or self._goal_handle is None:
            self.get_logger().info("Kein aktives Nav2-Goal zum Canceln.")
            return

        self.get_logger().info("Cancel des aktuellen Nav2-Goals angefordert...")
        if self.gui:
            self.gui.set_status("Cancelling Nav goal...")

        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self._cancel_done_callback)

    def _cancel_done_callback(self, future):
        try:
            _ = future.result()
            self.get_logger().info("Cancel-Request verarbeitet.")
        except Exception as e:
            self.get_logger().error(f"Fehler beim Canceln: {e}")

        self.goal_active = False
        self._goal_handle = None
        if self.gui:
            self.gui.set_status("Nav goal cancelled.")


class GoalGui(QtWidgets.QWidget):
    """
    Qt-GUI mit:
      - Logo oben
      - GroupBox für x, y, theta (deg) (nur Idle-Mission relevant)
      - Missions-Buttons (0..4)
      - Go / Stop
      - Status-Zeile
    """
    def __init__(self, node: Nav2GoalClient):
        super().__init__()
        self.node = node
        self.node.set_gui(self)

        self.init_ui()

        # QTimer, um regelmäßig rclpy.spin_once aufzurufen
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.spin_ros_once)
        self.timer.start(50)  # alle 50ms

    def init_ui(self):
        self.setWindowTitle("TB3 Navigation & Mission GUI")

        main_layout = QtWidgets.QVBoxLayout()

        # --- Logo oben einblenden ---
        logo_pixmap = QtGui.QPixmap(LOGO_PATH)
        if not logo_pixmap.isNull():
            logo_label = QtWidgets.QLabel()
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            # Breite begrenzen, damit es nicht zu groß wird
            scaled_logo = logo_pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            main_layout.addWidget(logo_label)

            # Fenster-Icon setzen
            self.setWindowIcon(QtGui.QIcon(LOGO_PATH))
        else:
            # Nur Log-Meldung, keine Exception – GUI soll auch ohne Logo laufen
            if self.node is not None:
                self.node.get_logger().warn(f"Logo not found at {LOGO_PATH}")

        # --- GroupBox für Zielpose (nur Idle sinnvoll) ---
        pose_group = QtWidgets.QGroupBox("Idle goal pose (map frame)")
        pose_layout = QtWidgets.QFormLayout()

        self.x_edit = QtWidgets.QLineEdit("0.0")
        self.y_edit = QtWidgets.QLineEdit("0.0")
        self.theta_edit = QtWidgets.QLineEdit("0.0")  # in Grad

        pose_layout.addRow("x [m]:", self.x_edit)
        pose_layout.addRow("y [m]:", self.y_edit)
        pose_layout.addRow("theta [deg]:", self.theta_edit)

        pose_group.setLayout(pose_layout)
        main_layout.addWidget(pose_group)

        main_layout.addSpacing(10)

        # --- Missions-Buttons ---
        challenge_group = QtWidgets.QGroupBox("Missions")
        chall_layout = QtWidgets.QHBoxLayout()

        self.btn_idle = QtWidgets.QPushButton("Idle (0) – Nav2")
        self.btn_idle.clicked.connect(lambda: self.on_mission_selected(0))
        chall_layout.addWidget(self.btn_idle)

        self.btn_m1 = QtWidgets.QPushButton("Mission 1")
        self.btn_m1.clicked.connect(lambda: self.on_mission_selected(1))
        chall_layout.addWidget(self.btn_m1)

        self.btn_m2 = QtWidgets.QPushButton("Mission 2")
        self.btn_m2.clicked.connect(lambda: self.on_mission_selected(2))
        chall_layout.addWidget(self.btn_m2)

        self.btn_m3 = QtWidgets.QPushButton("Mission 3")
        self.btn_m3.clicked.connect(lambda: self.on_mission_selected(3))
        chall_layout.addWidget(self.btn_m3)

        self.btn_m4 = QtWidgets.QPushButton("Mission 4")
        self.btn_m4.clicked.connect(lambda: self.on_mission_selected(4))
        chall_layout.addWidget(self.btn_m4)

        challenge_group.setLayout(chall_layout)
        main_layout.addWidget(challenge_group)

        # Aktuelle Mission anzeigen
        self.mission_label = QtWidgets.QLabel("Current mission (selected): Idle (0)")
        main_layout.addWidget(self.mission_label)

        # Abstand
        main_layout.addSpacing(10)

        # --- Button-Leiste ---
        button_layout = QtWidgets.QHBoxLayout()
        self.go_button = QtWidgets.QPushButton("Go")
        self.go_button.clicked.connect(self.on_go_clicked)
        button_layout.addWidget(self.go_button)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)

        # --- Status ---
        main_layout.addSpacing(10)
        self.status_label = QtWidgets.QLabel("Idle")
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

        # Anfangszustand
        self.update_input_states()

    # ------------------------------------------------------------------
    # UI-Callbacks
    # ------------------------------------------------------------------
    def on_go_clicked(self):
        """
        Go-Button:
        - Wenn Mission 0 (Idle) ausgewählt -> Nav2-Goal mit x,y,theta
        - Wenn Mission >0 -> Nodes starten + mission_id publish (über start_selected_mission)
        """
        if self.node.current_mission == 0:
            # Idle: Nav2-Goal schicken
            try:
                x = float(self.x_edit.text())
                y = float(self.y_edit.text())
                theta_deg = float(self.theta_edit.text())
            except ValueError:
                self.set_status("Ungültige Eingabe: bitte Zahlen eingeben.")
                return

            theta_rad = math.radians(theta_deg)
            self.set_status("Sending Nav goal (Idle)...")
            self.node.send_goal(x, y, theta_rad)
        else:
            # Mission starten
            self.node.start_selected_mission()

        self.update_input_states()

    def on_stop_clicked(self):
        """
        Stop-Button:
        - Bei aktiver Mission: mission_id=0 publishen (& ggf. Launch-Prozess killen)
        - Sonst (Idle/Nav2): Nav2-Goal abbrechen
        """
        if self.node.active_mission != 0:
            self.node.stop_active_mission()
        else:
            self.node.cancel_goal()

        self.update_input_states()

    def on_mission_selected(self, mission_id: int):
        """
        Wird aufgerufen, wenn einer der Missions-Buttons gedrückt wird.
        Setzt NUR die Mission im Node (Vorauswahl), publish passiert erst bei Go.
        """
        self.node.set_mission(mission_id)
        self.update_input_states()

    # ------------------------------------------------------------------
    # UI-State-Logik
    # ------------------------------------------------------------------
    def update_input_states(self):
        """
        Zentraler Ort, der UI-Zustand steuert.

        - Eingabe x,y,theta:
            erlaubt nur, wenn
              * Mission 0 (Idle) ausgewählt
              * kein Nav-Goal aktiv
        - Go:
            gesperrt, wenn Nav-Goal aktiv (für Idle) – sonst erlaubt
        - Stop:
            erlaubt, wenn
              * Nav-Goal aktiv ODER eine Mission aktiv ist
        """
        nav_active = self.node.goal_active
        active_mission = self.node.active_mission
        current_mission = self.node.current_mission

        can_edit_pose = (current_mission == 0) and (not nav_active)
        can_press_go = not nav_active  # für Missionen >0 ist nav_active eh False
        can_press_stop = nav_active or (active_mission != 0)

        self.x_edit.setEnabled(can_edit_pose)
        self.y_edit.setEnabled(can_edit_pose)
        self.theta_edit.setEnabled(can_edit_pose)

        self.go_button.setEnabled(can_press_go)
        self.stop_button.setEnabled(can_press_stop)

    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------
    def set_status(self, text: str):
        """Status-Text unten setzen."""
        self.status_label.setText(text)

    def set_mission_display(self, mission_id: int):
        """Anzeige für aktuelle Mission (Vorauswahl) updaten."""
        if mission_id == 0:
            txt = "Current mission (selected): Idle (0)"
        else:
            txt = f"Current mission (selected): {mission_id}"
        self.mission_label.setText(txt)

    def spin_ros_once(self):
        """Wird periodisch aufgerufen, um ROS-Callbacks zu verarbeiten."""
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
        # Flags können sich geändert haben -> UI aktualisieren
        self.update_input_states()

    def closeEvent(self, event):
        """Qt-Event: Fenster wird geschlossen -> Timer stoppen."""
        self.timer.stop()
        event.accept()


def main(args=None):
    rclpy.init(args=args)

    app = QtWidgets.QApplication(sys.argv)

    node = Nav2GoalClient()
    gui = GoalGui(node)
    gui.show()

    ret = 0
    try:
        ret = app.exec_()
    except KeyboardInterrupt:
        print("Ctrl+C pressed, shutting down GUI and ROS...")
    finally:
        gui.timer.stop()
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(ret)


if __name__ == '__main__':
    main()
