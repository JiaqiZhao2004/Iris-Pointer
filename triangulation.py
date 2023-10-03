import math


def find_distance(frame_height, y_min, y_max, theta=40 / 180 * math.pi):
    height = y_max - y_min
    head_length = 16
    h = head_length * (frame_height / height)
    d = 1 / (math.tan(theta / 2)) * (h / 2)
    return d


def transform_x_window(distance, screen_width=31, r=0.247, c=0.0879):
    return screen_width / (distance + r) * r + c
