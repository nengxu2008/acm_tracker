import cv2


def draw_text_lines(
    frame,
    lines,
    origin=(20, 40),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.6,
    thickness=2,
    color=(255, 255, 255),
    outline_color=(0, 0, 0),
    line_padding=10,
):
    """Draw multiple lines of text on a frame, with a contrasting outline.

    Pass outline_color=None to skip the outline.
    """
    x, y = origin
    text_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1]
    line_height = text_height + line_padding

    for i, line in enumerate(lines):
        pos = (x, y + i * line_height)
        if outline_color is not None:
            cv2.putText(
                frame, line, pos, font, scale, outline_color, thickness + 2, cv2.LINE_AA
            )
        cv2.putText(frame, line, pos, font, scale, color, thickness, cv2.LINE_AA)

    return frame


def draw_stats(frame, frame_idx, fps, iou, center_error_text):
    return draw_text_lines(
        frame,
        [
            f"Frame: {frame_idx} | FPS: {fps:.1f}",
            f"IoU: {iou:.3f} | Ct err: {center_error_text}",
        ],
    )