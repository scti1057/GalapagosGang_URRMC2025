#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission3BehaviorTree für Turtlebot3 + Nav2.

Aufgabe:
- Lauscht auf /mission (std_msgs/String).
- Wenn Mission 3 aktiv wird, läuft folgender Behavior Tree:

  Root (Sequence)
    1) WaitForMission3
    2) DriveInBoxPhase
         - aktiviert tb3_drive_in_box über mission3/drive_in_box_enable
         - wartet auf /start_exploration == True
    3) ExploreBoxPhase
         - aktiviert frontier_explorer_node über mission3/frontier_enable
         - wartet auf mission3/done == True

Hinweis:
- tb3_drive_in_box muss ein Bool-Topic mission3/drive_in_box_enable abonnieren.
- frontier_explorer_node muss Bool-Topic mission3/frontier_enable + mission3/done verwenden.
"""

from enum import Enum, auto

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class BTNode:
    def tick(self) -> Status:
        raise NotImplementedError


class Sequence(BTNode):
    """Einfacher BT-Sequence-Knoten."""

    def __init__(self, children):
        self.children = list(children)
        self.current_index = 0

    def tick(self) -> Status:
        # Wenn alle Kinder fertig -> SUCCESS
        while self.current_index < len(self.children):
            child = self.children[self.current_index]
            status = child.tick()

            if status == Status.SUCCESS:
                # nächstes Kind
                self.current_index += 1
                continue

            if status == Status.RUNNING:
                return Status.RUNNING

            if status == Status.FAILURE:
                # Sequence scheitert -> von vorne
                self.current_index = 0
                return Status.FAILURE

        return Status.SUCCESS


# ---------------------------------------------------------------------------
# Blatt-Knoten
# ---------------------------------------------------------------------------

class WaitForMission3(BTNode):
    """Wartet, bis /mission Mission 3 signalisiert."""

    def __init__(self, ros_node: Node):
        self.node = ros_node
        self._first_log = False

    def tick(self) -> Status:
        if not self._first_log:
            self.node.get_logger().info("[BT] Warte auf Mission 3 über /mission ...")
            self._first_log = True

        if self.node.mission3_active:
            # Nur einmal loggen
            if not self.node._mission3_logged:
                self.node.get_logger().info("[BT] Mission 3 erkannt -> starte Challenge-3-BT.")
                self.node._mission3_logged = True
            return Status.SUCCESS

        # Noch nicht Mission 3 -> Sequence bleibt im ersten Knoten hängen
        return Status.RUNNING


class DriveInBoxPhase(BTNode):
    """
    Phase 1: Bot fährt in den Kasten.

    - setzt mission3/drive_in_box_enable = True
    - wartet, bis /start_exploration == True (von tb3_drive_in_box, wenn Nav2-Ziel im Kasten erreicht)
    """

    def __init__(self, ros_node: Node):
        self.node = ros_node
        self.started = False
        self.done = False

    def tick(self) -> Status:
        if not self.started:
            self.node.get_logger().info("[BT] Phase 1: DriveInBox aktivieren.")
            self.node.enable_drive_in_box(True)
            self.started = True

        if self.done:
            return Status.SUCCESS

        # Warten auf Trigger von tb3_drive_in_box
        if self.node.start_exploration_triggered:
            self.node.get_logger().info("[BT] Phase 1 fertig: /start_exploration empfangen.")
            self.node.enable_drive_in_box(False)
            self.done = True
            return Status.SUCCESS

        return Status.RUNNING


class ExploreBoxPhase(BTNode):
    """
    Phase 2: Frontier-Explorer im Kasten aktiv, bis Exit gefunden.

    - setzt mission3/frontier_enable = True
    - wartet auf mission3/done == True
    """

    def __init__(self, ros_node: Node):
        self.node = ros_node
        self.started = False

    def tick(self) -> Status:
        if not self.started:
            self.node.get_logger().info("[BT] Phase 2: Frontier-Exit-Explorer aktivieren.")
            self.node.enable_frontier(True)
            self.started = True

        if self.node.challenge3_done:
            self.node.get_logger().info("[BT] Phase 2 fertig: mission3/done empfangen.")
            self.node.enable_frontier(False)
            return Status.SUCCESS

        return Status.RUNNING


# ---------------------------------------------------------------------------
# ROS2-Wrapper für BT
# ---------------------------------------------------------------------------

class Mission3BtNode(Node):
    def __init__(self):
        super().__init__("mission3_bt")

        # ---- State ----
        self.mission3_active: bool = False
        self._mission3_logged: bool = False

        self.start_exploration_triggered: bool = False
        self.challenge3_done: bool = False

        self.drive_in_box_enabled: bool = False
        self.frontier_enabled: bool = False

        # ---- Subscriptions ----
        self.mission_sub = self.create_subscription(
            String,
            "/mission",
            self.mission_cb,
            10,
        )

        # kommt aus tb3_drive_in_box, wenn Nav2-Goal im Kasten erreicht ist
        self.start_exploration_sub = self.create_subscription(
            Bool,
            "start_exploration",   # ggf. "/start_exploration" je nach Launch
            self.start_exploration_cb,
            10,
        )

        # kommt aus frontier_explorer_node, wenn Exit erreicht ist
        self.done_sub = self.create_subscription(
            Bool,
            "mission3/done",
            self.challenge3_done_cb,
            10,
        )

        # ---- Publisher, um die beiden Nodes zu "steuern" ----
        self.drive_in_box_enable_pub = self.create_publisher(
            Bool,
            "mission3/drive_in_box_enable",
            10,
        )
        self.frontier_enable_pub = self.create_publisher(
            Bool,
            "mission3/frontier_enable",
            10,
        )

        # ---- Behavior Tree aufbauen ----
        self.root = Sequence(
            [
                WaitForMission3(self),
                DriveInBoxPhase(self),
                ExploreBoxPhase(self),
            ]
        )

        # Timer zum Ticken des BT
        self.bt_timer = self.create_timer(0.2, self.bt_tick)

        self.get_logger().info("Mission3BehaviorTree Node gestartet.")

    # ----------------- Callbacks -----------------

    def mission_cb(self, msg: String):
        s = msg.data.strip().lower()
        active = s in ("3", "mission3", "challenge3", "challenge_3")
        if active != self.mission3_active:
            self.mission3_active = active
            self.get_logger().info(f"[BT] /mission='{msg.data}' -> mission3_active={self.mission3_active}")

    def start_exploration_cb(self, msg: Bool):
        # wird von drive_in_box gesetzt, wenn Ziel im Kasten erreicht
        if msg.data and not self.start_exploration_triggered:
            self.get_logger().info(
                "[BT] /start_exploration == True empfangen -> Phase 1 aus, Phase 2 an."
            )

            # Sofort umschalten:
            # 1) DriveInBox deaktivieren
            self.enable_drive_in_box(False)

            # 2) Frontier-Explorer aktivieren
            self.enable_frontier(True)

        # Flag setzen, damit der BT-Node DriveInBoxPhase als "fertig" erkennt
        self.start_exploration_triggered = msg.data


    def challenge3_done_cb(self, msg: Bool):
        if msg.data and not self.challenge3_done:
            self.get_logger().info("[BT] mission3/done == True empfangen.")
        self.challenge3_done = msg.data

    # ----------------- Aktoren für die Nodes -----------------

    def enable_drive_in_box(self, enable: bool):
        if self.drive_in_box_enabled == enable:
            return
        self.drive_in_box_enabled = enable
        msg = Bool()
        msg.data = enable
        self.drive_in_box_enable_pub.publish(msg)
        self.get_logger().info(f"[BT] mission3/drive_in_box_enable -> {enable}")

    def enable_frontier(self, enable: bool):
        if self.frontier_enabled == enable:
            return
        self.frontier_enabled = enable
        msg = Bool()
        msg.data = enable
        self.frontier_enable_pub.publish(msg)
        self.get_logger().info(f"[BT] mission3/frontier_enable -> {enable}")

    # ----------------- BT-Ticker -----------------

    def bt_tick(self):
        status = self.root.tick()
        # Debug-Log nur grob
        if status == Status.SUCCESS:
            self.get_logger().info("[BT] Mission 3 Behavior Tree vollständig abgeschlossen.")
            # Optional: Timer stoppen
            try:
                self.bt_timer.cancel()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = Mission3BtNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
