#!/usr/bin/env python3
import math
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class GoToGoal(Node):
    """
    Satu node mandiri (tidak perlu node lain) untuk menggerakkan robot ke koordinat:
      1. Posisi robot dihitung sendiri dari /joint_states (forward kinematics)
      2. Go-to-goal         : pose + tujuan -> kecepatan body (v, w)
      3. Inverse kinematics : (v, w) -> kecepatan roda kiri & kanan
    Koordinat tujuan diketik langsung di terminal node ini.
    """

    def __init__(self):
        super().__init__('go_to_goal')

        # Parameter fisik robot
        self.r = 0.03
        self.L = 0.17
        self.left_joint = 'base_left_wheel_joint'
        self.right_joint = 'base_right_wheel_joint'

        # Gain dan batas kecepatan (boleh disetel)
        self.k_lin = 0.5
        self.k_ang = 1.5
        self.v_max = 0.2
        self.w_max = 1.5
        self.goal_tol = 0.03
        self.turn_first = 0.5   # rad, jika error sudut lebih besar -> putar dulu

        # Pose robot (mulai dari 0, 0, 0 saat node dijalankan)
        self.x = self.y = self.theta = 0.0
        self.prev_left = None
        self.prev_right = None
        self.have_pose = False
        self.goal = None

        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.left_pub = self.create_publisher(Float64, '/left_wheel/command', 10)
        self.right_pub = self.create_publisher(Float64, '/right_wheel/command', 10)

        self.create_timer(0.05, self.control_loop)  # 20 Hz

        threading.Thread(target=self.input_loop, daemon=True).start()

    # ---------- 1. Forward kinematics (posisi dari putaran roda) ----------
    def joint_callback(self, msg):
        if self.left_joint not in msg.name or self.right_joint not in msg.name:
            self.get_logger().warn(
                f'Joint tidak ditemukan. Tersedia: {list(msg.name)}',
                throttle_duration_sec=2.0)
            return

        left = msg.position[msg.name.index(self.left_joint)]
        right = msg.position[msg.name.index(self.right_joint)]

        if self.prev_left is None:
            self.prev_left, self.prev_right = left, right
            self.have_pose = True
            return

        d_left = self.r * (left - self.prev_left)
        d_right = self.r * (right - self.prev_right)
        self.prev_left, self.prev_right = left, right

        ds = (d_right + d_left) / 2.0
        dtheta = (d_right - d_left) / self.L

        self.x += ds * math.cos(self.theta + dtheta / 2.0)
        self.y += ds * math.sin(self.theta + dtheta / 2.0)
        self.theta = math.atan2(math.sin(self.theta + dtheta),
                                math.cos(self.theta + dtheta))

    # ---------- 2. Go-to-goal ----------
    def control_loop(self):
        if self.goal is None or not self.have_pose:
            return

        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        distance = math.hypot(dx, dy)

        if distance < self.goal_tol:
            self.send_wheels(0.0, 0.0)
            print(f'\nSampai di tujuan (x={self.x:.2f}, y={self.y:.2f})')
            print('tujuan> ', end='', flush=True)
            self.goal = None
            return

        angle_to_goal = math.atan2(dy, dx)
        err = math.atan2(math.sin(angle_to_goal - self.theta),
                         math.cos(angle_to_goal - self.theta))

        w = max(-self.w_max, min(self.w_max, self.k_ang * err))
        v = 0.0 if abs(err) > self.turn_first else min(self.v_max, self.k_lin * distance)

        self.send_wheels(v, w)

    # ---------- 3. Inverse kinematics ----------
    def send_wheels(self, v, w):
        w_left = (v - w * self.L / 2.0) / self.r
        w_right = (v + w * self.L / 2.0) / self.r

        left_msg = Float64()
        right_msg = Float64()
        left_msg.data = float(w_left)
        right_msg.data = float(w_right)
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    # ---------- Input koordinat dari terminal ----------
    def input_loop(self):
        print('Masukkan koordinat tujuan (meter), format: x y  (contoh: 1.0 0.5)')
        print('Ketik s untuk berhenti, Ctrl+C untuk keluar.')
        while rclpy.ok():
            try:
                text = input('tujuan> ').strip()
            except EOFError:
                return
            if text.lower() == 's':
                self.goal = None
                self.send_wheels(0.0, 0.0)
                print('Robot berhenti.')
                continue
            try:
                x, y = map(float, text.split())
            except ValueError:
                print('Format salah. Contoh: 1.0 0.5')
                continue
            self.goal = (x, y)
            print(f'Menuju x={x}, y={y}')


def main(args=None):
    rclpy.init(args=args)
    node = GoToGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send_wheels(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()