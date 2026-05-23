from math import hypot, pi
import cv2
import numpy as np

WINDOW_NAME = "Falling Ball - Camera"
BALL_START_Y = 40

input_mode = "camera"
camera_index = 0
projector_fullscreen = False

ball_radius = 15
gravity = 0.5
friction = 0.98
max_speed = 15.0

min_platform_length = 50
adaptive_threshold_block_size = 21
adaptive_threshold_c = 5
hough_threshold = 50
max_line_gap = 10
platform_line_thickness = 3

def open_camera():
    if input_mode != "camera":
        return None
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Error: could not open camera index {camera_index}.")
        return None
    return camera

def read_frame(camera):
    return camera.read()

def setup_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if projector_fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def get_ball_start_position(frame):
    height, width = frame.shape[:2]
    return [width // 2, min(BALL_START_Y, height // 2)]

def reset_ball(frame):
    return get_ball_start_position(frame), [0.0, 0.0], False

def is_ball_out_of_screen(ball_pos, frame):
    height, width = frame.shape[:2]
    x, y = ball_pos
    return (
        x + ball_radius < 0
        or x - ball_radius > width
        or y + ball_radius < 0
        or y - ball_radius > height
    )

def detect_platforms(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_threshold_block_size,
        adaptive_threshold_c,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(
        mask,
        1,
        pi / 180,
        threshold=hough_threshold,
        minLineLength=min_platform_length,
        maxLineGap=max_line_gap,
    )
    platforms = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if hypot(x2 - x1, y2 - y1) < min_platform_length:
                continue
            platforms.append(((int(x1), int(y1)), (int(x2), int(y2))))
    return platforms

def closest_point_on_segment(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    segment_x = bx - ax
    segment_y = by - ay
    segment_length_squared = segment_x * segment_x + segment_y * segment_y
    if segment_length_squared == 0:
        return float(ax), float(ay)
    t = ((px - ax) * segment_x + (py - ay) * segment_y) / segment_length_squared
    t = max(0.0, min(1.0, t))
    return ax + t * segment_x, ay + t * segment_y

def limit_speed(velocity):
    speed = hypot(velocity[0], velocity[1])
    if speed <= max_speed:
        return velocity
    scale = max_speed / speed
    return [velocity[0] * scale, velocity[1] * scale]

def handle_collisions(ball_pos, ball_velocity, platforms):
    for start_point, end_point in platforms:
        segment_x = end_point[0] - start_point[0]
        segment_y = end_point[1] - start_point[1]
        segment_length = hypot(segment_x, segment_y)
        if segment_length == 0:
            continue
        tangent_x = segment_x / segment_length
        tangent_y = segment_y / segment_length
        closest_x, closest_y = closest_point_on_segment(
            ball_pos,
            start_point,
            end_point,
        )
        normal_x = ball_pos[0] - closest_x
        normal_y = ball_pos[1] - closest_y
        distance = hypot(normal_x, normal_y)
        if distance > ball_radius:
            continue
        if distance == 0:
            normal_x = -tangent_y
            normal_y = tangent_x
            if normal_y > 0:
                normal_x = -normal_x
                normal_y = -normal_y
        else:
            normal_x /= distance
            normal_y /= distance
        overlap = ball_radius - distance
        if overlap > 0:
            ball_pos[0] += normal_x * overlap
            ball_pos[1] += normal_y * overlap
        tangent_speed = ball_velocity[0] * tangent_x + ball_velocity[1] * tangent_y
        gravity_along_tangent = gravity * tangent_y
        tangent_speed = (tangent_speed + gravity_along_tangent) * friction
        ball_velocity[0] = tangent_x * tangent_speed
        ball_velocity[1] = tangent_y * tangent_speed
    return ball_pos, limit_speed(ball_velocity)

def update_physics(ball_pos, ball_velocity, platforms, frame):
    ball_velocity[1] += gravity
    ball_velocity = limit_speed(ball_velocity)
    ball_pos[0] += ball_velocity[0]
    ball_pos[1] += ball_velocity[1]
    ball_pos, ball_velocity = handle_collisions(
        ball_pos,
        ball_velocity,
        platforms,
    )
    if is_ball_out_of_screen(ball_pos, frame):
        return reset_ball(frame)
    return ball_pos, ball_velocity, True

def main():
    camera = open_camera()
    if input_mode == "camera" and camera is None:
        return
    setup_window()
    ok, camera_frame = read_frame(camera)
    if not ok:
        print("Error: could not read frame from webcam.")
        return
    height, width = camera_frame.shape[:2]
    black_bg = np.zeros((height, width, 3), dtype=np.uint8)
    ball_pos = get_ball_start_position(black_bg)
    ball_velocity = [0.0, 0.0]
    simulation_started = False
    show_debug = False

    try:
        while True:
            ok, camera_frame = read_frame(camera)
            if not ok:
                print("Error: could not read frame from webcam.")
                break

            platforms = detect_platforms(camera_frame) if show_debug else []

            if simulation_started:
                ball_pos, ball_velocity, simulation_started = update_physics(
                    ball_pos,
                    ball_velocity,
                    platforms,
                    black_bg,
                )

            frame_to_show = black_bg.copy()
            if show_debug:
                for start_point, end_point in platforms:
                    cv2.line(
                        frame_to_show,
                        start_point,
                        end_point,
                        (0, 255, 0),
                        platform_line_thickness,
                    )

            ball_center = (int(ball_pos[0]), int(ball_pos[1]))
            cv2.circle(frame_to_show, ball_center, ball_radius, (0, 0, 255), -1)
            cv2.imshow(WINDOW_NAME, frame_to_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                simulation_started = True
            if key == ord("r"):
                ball_pos, ball_velocity, simulation_started = reset_ball(black_bg)
            if key == ord("d"):
                show_debug = not show_debug

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
