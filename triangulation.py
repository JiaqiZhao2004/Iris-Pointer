import math


class Triangulation(object):
    def __init__(self, frame_w, frame_h, theta_x, theta_y, real_head_width=16):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.real_head_width = real_head_width

    def find_distance(self, y_min, y_max):
        h = y_max - y_min
        fov_h = self.real_head_width * (self.frame_h / h)
        return 1 / (math.tan(self.theta_y / 2)) * (fov_h / 2)

    def find_eye_displacement(self, eye_center_x, eye_center_y, y_min, y_max):
        h = y_max - y_min
        fov_h = self.real_head_width * (self.frame_h / h)
        fov_w = fov_h * self.frame_w / self.frame_h
        vertical = (0.5 - eye_center_y / self.frame_h) * fov_h
        horizontal = (0.5 - eye_center_x / self.frame_w) * fov_w
        return round(horizontal), round(vertical)


def transform_x_window(distance, screen_width=31, r=0.247, c=0.0879):
    return screen_width / (distance + r) * r + c
