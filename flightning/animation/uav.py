import numpy as np
from matplotlib.colors import to_rgba
from .basic import Line, Arrow, Plate


class Uav:
    """
    Draws a quadrotor at a given position and attitude.
    """

    def __init__(self, ax, scale=1.0, color="k"):
        self.ax = ax

        # body axes
        self.b1 = np.array([0.2, 0.0, 0.0]) * scale
        self.b2 = np.array([0.0, 0.2, 0.0]) * scale
        self.b3 = np.array([0.0, 0.0, 0.2]) * scale

        # motor offsets in body frame (FR, BL, BR, FL)
        self.fr = np.array([0.16, -0.16, 0.0]) * scale
        self.bl = np.array([-0.16, 0.16, 0.0]) * scale
        self.br = np.array([-0.16, -0.16, 0.0]) * scale
        self.fl = np.array([0.16, 0.16, 0.0]) * scale

        # body & motors
        self.body = Plate(self.ax, r=0.1 * scale, c=color)
        self.motor1 = Plate(self.ax, r=0.1 * scale, c=color)
        self.motor2 = Plate(self.ax, r=0.1 * scale, c=color)
        self.motor3 = Plate(self.ax, r=0.1 * scale, c=color)
        self.motor4 = Plate(self.ax, r=0.1 * scale, c=color)
        self.motor_alpha = 1
        self.arm_fr = Line(self.ax, c=color)
        self.arm_bl = Line(self.ax, c=color)
        self.arm_br = Line(self.ax, c=color)
        self.arm_fl = Line(self.ax, c=color)

        # triad
        self.arrow_b1 = Arrow(self.ax, c="r")
        self.arrow_b2 = Arrow(self.ax, c="g")
        self.arrow_b3 = Arrow(self.ax, c="b")

    @staticmethod
    def _apply_facecolor(plate: Plate, color, alpha: float):
        rgba = to_rgba(color, alpha=alpha)
        plate.collection.set_facecolor([rgba])

    def _set_motor_colors(self, motor_colors):
        self._apply_facecolor(self.motor1, motor_colors[0], self.motor_alpha)
        self._apply_facecolor(self.motor2, motor_colors[1], self.motor_alpha)
        self._apply_facecolor(self.motor3, motor_colors[2], self.motor_alpha)
        self._apply_facecolor(self.motor4, motor_colors[3], self.motor_alpha)

    def draw_at(self, x=np.array([0.0, 0.0, 0.0]), R=np.eye(3), motor_colors=None):
        x = np.asarray(x).reshape(3)
        R = np.asarray(R).reshape(3, 3)

        if motor_colors is not None:
            self._set_motor_colors(motor_colors)

        # center marker
        self.body.draw_at(x, R)

        # rotor plates
        self.motor1.draw_at(x + R @ self.fr, R)
        self.motor2.draw_at(x + R @ self.bl, R)
        self.motor3.draw_at(x + R @ self.br, R)
        self.motor4.draw_at(x + R @ self.fl, R)

        # body axes (as lines)
        self.arrow_b1.draw_from_to(x, R @ self.b1)
        self.arrow_b2.draw_from_to(x, R @ self.b2)
        self.arrow_b3.draw_from_to(x, R @ self.b3)

        # arms
        self.arm_fr.draw_from_to(x, x + R @ self.fr)
        self.arm_bl.draw_from_to(x, x + R @ self.bl)
        self.arm_br.draw_from_to(x, x + R @ self.br)
        self.arm_fl.draw_from_to(x, x + R @ self.fl)
