#!/usr/bin/env python3

import sys
import math

from PyQt5 import QtWidgets, QtCore

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class Nav2GoalClient(Node):
    def __init__(self):
        super().__init__('nav2_goal_client')
        # Action-Client für Nav2
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_active = False
        self._goal_handle = None

        # Lane-Following Status
        self.lane_following_running = False
        self.lane_pub = self.create_publisher(Bool, 'lane_following_enabled', 10)

        self.gui = None  # Referenz auf GUI, wird später gesetzt

    def set_gui(self, gui):
        self.gui = gui

    # ----------------- Lane Following -----------------

    def _publish_lane_state(self):
        msg = Bool()
        msg.data = self.lane_following_running
        self.lane_pub.publish(msg)

    def start_lane_following(self):
        """Lane-Following starten (nur, wenn noch nicht läuft)."""
        if self.lane_following_running:
            self.get_logger().info("Lane following läuft bereits.")
            if self.gui:
                self.gui.set_status("Lane following already running.")
            return

        self.lane_following_running = True
        self._publish_lane_state()
        self.get_logger().info("Lane following ENABLED.")
        if self.gui:
            self.gui.set_status("Lane following ENABLED.")

    def stop_lane_following(self):
        """Lane-Following stoppen (falls aktiv)."""
        if not self.lane_following_running:
            return

        self.lane_following_running = False
        self._publish_lane_state()
        self.get_logger().info("Lane following DISABLED.")
        if self.gui:
            self.gui.set_status("Lane following DISABLED.")

    # ----------------- Nav2 Goal Handling -----------------

    def send_goal(self, x: float, y: float, yaw: float):
        """Goal an Nav2 schicken (x, y in m, yaw in rad im map-Frame)."""
        if self.lane_following_running:
            self.get_logger().warn("Lane following aktiv, Goal wird nicht gesendet.")
            if self.gui:
                self.gui.set_status("Lane following running, cannot send goal.")
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
        if self.gui and not self.lane_following_running:
            # Status wird bei Lane-Cancel separat gesetzt
            self.gui.set_status("Nav goal cancelled.")


class GoalGui(QtWidgets.QWidget):
    """
    Qt-GUI mit:
      - GroupBox für x, y, theta (deg)
      - Lane-Following-Checkbox (Moduswahl)
      - Go / Stop
      - Status-Zeile
    """
    def __init__(self, node: Nav2GoalClient):
        super().__init__()
        self.node = node
        self.node.set_gui(self)

        # UI-Zustand
        self.lane_mode_selected = False   # Checkbox-Zustand

        self.init_ui()

        # QTimer, um regelmäßig rclpy.spin_once aufzurufen
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.spin_ros_once)
        self.timer.start(50)  # alle 50ms

    def init_ui(self):
        self.setWindowTitle("TB3 Navigation GUI")

        main_layout = QtWidgets.QVBoxLayout()

        # --- GroupBox für Zielpose ---
        pose_group = QtWidgets.QGroupBox("Goal pose (map frame)")
        pose_layout = QtWidgets.QFormLayout()

        self.x_edit = QtWidgets.QLineEdit("0.0")
        self.y_edit = QtWidgets.QLineEdit("0.0")
        self.theta_edit = QtWidgets.QLineEdit("0.0")  # in Grad

        pose_layout.addRow("x [m]:", self.x_edit)
        pose_layout.addRow("y [m]:", self.y_edit)
        pose_layout.addRow("theta [deg]:", self.theta_edit)

        pose_group.setLayout(pose_layout)
        main_layout.addWidget(pose_group)

        # Abstand zur Lane-Checkbox
        main_layout.addSpacing(10)

        # --- Lane-Following Checkbox ---
        self.lane_checkbox = QtWidgets.QCheckBox("Lane following")
        self.lane_checkbox.stateChanged.connect(self.on_lane_checkbox_changed)
        main_layout.addWidget(self.lane_checkbox)

        # Abstand zu Buttons
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

    # ----------------- UI-Callbacks -----------------

    def on_go_clicked(self):
        """
        Go-Button:
        - Wenn Lane-Mode NICHT ausgewählt -> Nav2-Goal mit x,y,theta
        - Wenn Lane-Mode ausgewählt -> Lane-Following starten
        """
        if self.lane_mode_selected:
            # Lane following erst hier starten
            if not self.node.lane_following_running:
                self.node.start_lane_following()
            else:
                self.set_status("Lane following already running.")
        else:
            # Normales Nav2-Goal
            try:
                x = float(self.x_edit.text())
                y = float(self.y_edit.text())
                theta_deg = float(self.theta_edit.text())
            except ValueError:
                self.set_status("Ungültige Eingabe: bitte Zahlen eingeben.")
                return

            theta_rad = math.radians(theta_deg)
            self.set_status("Sending Nav goal...")
            self.node.send_goal(x, y, theta_rad)

        self.update_input_states()

    def on_stop_clicked(self):
        """
        Stop-Button:
        - bricht Nav2-Goal ab
        - stoppt Lane-Following (falls aktiv)
        - nimmt Lane-Checkbox raus
        """
        self.node.cancel_goal()
        self.node.stop_lane_following()

        if self.lane_mode_selected:
            # Checkbox visuell auch zurücksetzen (triggert on_lane_checkbox_changed)
            self.lane_checkbox.setChecked(False)

        self.update_input_states()

    def on_lane_checkbox_changed(self, state):
        """
        Lane-Mode an/aus:
        - an: Positions-Eingabe sperren, Go bleibt aktiv (Go startet Lane-Following)
        - aus: Falls Lane-Following läuft -> stoppen
        """
        checked = state == QtCore.Qt.Checked
        self.lane_mode_selected = checked

        if checked:
            if self.node.lane_following_running:
                # theoretisch selten
                self.set_status("Lane following already running.")
            else:
                self.set_status("Lane mode selected. Press Go to start lane following.")
        else:
            # Checkbox aus -> Lane-Following sicher aus
            if self.node.lane_following_running:
                self.node.stop_lane_following()
            self.set_status("Lane mode off.")

        self.update_input_states()

    # ----------------- UI-State-Logik -----------------

    def update_input_states(self):
        """
        Zentraler Ort, der UI-Zustand steuert.

        - Eingabe x,y,theta:
            erlaubt nur, wenn
              * kein Nav-Goal aktiv
              * kein Lane-Following aktiv
              * Lane-Mode NICHT ausgewählt
        - Go:
            erlaubt nur, wenn
              * kein Nav-Goal aktiv
              * kein Lane-Following aktiv
            (Lane-Mode bestimmt nur, WAS Go tut)
        - Stop:
            erlaubt, wenn
              * Nav-Goal aktiv ODER Lane-Following aktiv
        """
        nav_active = self.node.goal_active
        lane_running = self.node.lane_following_running

        can_edit_pose = (not nav_active) and (not lane_running) and (not self.lane_mode_selected)
        can_press_go = (not nav_active) and (not lane_running)
        can_press_stop = nav_active or lane_running

        self.x_edit.setEnabled(can_edit_pose)
        self.y_edit.setEnabled(can_edit_pose)
        self.theta_edit.setEnabled(can_edit_pose)

        self.go_button.setEnabled(can_press_go)
        self.stop_button.setEnabled(can_press_stop)

    # ----------------- Hilfsfunktionen -----------------

    def set_status(self, text: str):
        """Status-Text unten setzen."""
        self.status_label.setText(text)

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
