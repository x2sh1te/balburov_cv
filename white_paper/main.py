import cv2
import numpy as np
import zmq

OVERLAY_STRING = "FDS"
TYPEFACE = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 3.0
TEXT_COLOR = (0, 0, 255)  # BGR
THICK = 5
STREAM_URL = "tcp://84.237.21.36:6002"
DOC_WIDTH, DOC_HEIGHT = 400, 565


def detect_document_bounds(image):

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(smoothed, 50, 150)

    found_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found_contours:
        return None

    found_contours = sorted(found_contours, key=cv2.contourArea, reverse=True)
    min_area = image.shape[0] * image.shape[1] * 0.05

    for contour in found_contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(polygon) == 4 and cv2.contourArea(polygon) > min_area:
            return polygon.reshape(4, 2).astype(np.float32)

    return None


def sort_points(points):

    ordered = np.zeros((4, 2), dtype=np.float32)
    axis_sum = points.sum(axis=1)

    ordered[0] = points[np.argmin(axis_sum)]
    ordered[2] = points[np.argmax(axis_sum)]

    axis_diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(axis_diff)]
    ordered[3] = points[np.argmax(axis_diff)]

    return ordered


def generate_overlay_graphic(w, h, message):

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(message, TYPEFACE, SCALE, THICK)

    pos_x = (w - text_w) // 2
    pos_y = (h + text_h) // 2

    cv2.putText(canvas, message, (pos_x, pos_y), TYPEFACE, SCALE, TEXT_COLOR, THICK, cv2.LINE_AA)
    return canvas


def main():
    ctx = zmq.Context()
    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
    sub_socket.connect(STREAM_URL)

    window_name = 'Video Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    frame_counter = 0
    ref_points = np.array([[0, 0], [DOC_WIDTH, 0], [DOC_WIDTH, DOC_HEIGHT], [0, DOC_HEIGHT]], dtype=np.float32)
    overlay_mat = generate_overlay_graphic(DOC_WIDTH, DOC_HEIGHT, OVERLAY_STRING)

    while True:
        packet = sub_socket.recv()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1
        current_img = cv2.imdecode(np.frombuffer(packet, np.uint8), -1)

        if current_img is None:
            continue

        raw_corners = detect_document_bounds(current_img)

        if raw_corners is not None:
            target_points = sort_points(raw_corners)

            transform_matrix = cv2.getPerspectiveTransform(ref_points, target_points)
            transformed_overlay = cv2.warpPerspective(overlay_mat, transform_matrix,
                                                      (current_img.shape[1], current_img.shape[0]))

            overlay_gray = cv2.cvtColor(transformed_overlay, cv2.COLOR_BGR2GRAY)
            _, threshold_mask = cv2.threshold(overlay_gray, 254, 255, cv2.THRESH_BINARY_INV)

            background = cv2.bitwise_and(current_img, current_img, mask=cv2.bitwise_not(threshold_mask))
            foreground = cv2.bitwise_and(transformed_overlay, transformed_overlay, mask=threshold_mask)
            current_img = cv2.add(background, foreground)

        cv2.putText(current_img, f"Processed: {frame_counter}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 2)
        cv2.imshow(window_name, current_img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()