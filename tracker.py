import cv2
import numpy as np
import os, glob, csv
from ultralytics.utils.ops import xyxyxyxy2xywhr, xywhr2xyxyxyxy

def generate_rois(frame, obb, max_move=100, padding=10):
    """
    Args:
        frame: Original image (H, W, 3)
        obb: [cx, cy, w, h, theta], theta in radians
        max_move: Maximum center displacement along either axis
        padding: Extra space around the object

    Returns:
        Two cropped, rotated strips as images.
    """

    cx, cy, w, h, theta = map(float, obb)

    # Width/height of the search strips
    length = max(1, int(np.ceil(2 * max_move + max(w, h) + 2 * padding)))
    thickness = max(1, int(np.ceil(max(w, h) + 2 * padding)))

    rois = []
    transforms = []

    for angle in [theta, theta + np.pi / 2]:
        # Rotate the image so the search axis becomes horizontal
        M = cv2.getRotationMatrix2D(
            (cx, cy),
            np.degrees(angle),
            1.0
        )

        # Translate the rotated image so the object center appears at the center of the output crop.
        M[0, 2] += length / 2 - cx
        M[1, 2] += thickness / 2 - cy

        crop = cv2.warpAffine(
            frame,
            M,
            (length, thickness),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        rois.append(crop)
        transforms.append(M)
    return rois, transforms





from detector import detector
class tracker:
    def __init__(self, first_frame, init_bbox):
        self.first_frame = first_frame
        self.bbox = init_bbox

    def update(self, next_frame):
        """Select the highest-confidence OBB from the two search crops."""
        self.rois, transforms = generate_rois(next_frame, self.bbox)
        results = detector(self.rois)

        best_conf = -1.0
        best_corners = None

        # i identifies the crop; j identifies an object within that crop.
        for i, result in enumerate(results):
            if result.obb is None:
                continue

            for j, confidence in enumerate(result.obb.conf):
                confidence = float(confidence)
                if confidence <= best_conf:
                    continue

                crop_corners = result.obb.xyxyxyxy[j].cpu().numpy()
                best_corners = cv2.transform(
                    crop_corners[None].astype(np.float32),
                    cv2.invertAffineTransform(transforms[i]),
                )[0]
                best_conf = confidence

        if best_corners is None:
            self.success = False
            return self.success, self.bbox

        # Store five OBB values, not the four corner points.
        frame_obb = xyxyxyxy2xywhr(best_corners[None])[0]
        self.bbox = tuple(float(v) for v in frame_obb)
        self.success = True
        return self.success, self.bbox


def tracking(sequence_folders, output_dir):
    """Track every sequence folder and write its CSV and video outputs."""
    os.makedirs(output_dir, exist_ok=True)

    for seq_folder in sequence_folders:
        if not os.path.isdir(seq_folder):
            continue

        # create file name
        folder_name = os.path.basename(seq_folder)
        csv_filename = f"tracker_log_{folder_name}.csv"
        csv_file_path = os.path.join(output_dir, csv_filename)
        vid_filename = f"tracker_log_{folder_name}.mp4"
        vid_file_path = os.path.join(output_dir, vid_filename)

        # Get ground-truth bounding box
        gt_path = os.path.join(seq_folder, "groundtruth.txt")

        # Get sorted frames for THIS specific video sequence
        frame_paths = sorted(glob.glob(os.path.join(seq_folder, "*.jpg")))

        # Parse initial ground truth bounding box (line 1 of groundtruth.txt)
        with open(gt_path, "r") as f:
                gt_lines = [
                    tuple(float(v) for v in line.strip().split(","))
                    for line in f
                    if line.strip()
                ]

        # Initialize bbox
        x, y, w, h = gt_lines[0]
        bbox_init = (x + w / 2, y + h / 2, w, h, 0.0)

        # Read the first frame to initialize ROI selection
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            print("Failed to load the first image frame.")
            continue

        # Save Video
        #   GET FRAME DIMENSIONS (Width, Height)
        height, width, _ = first_frame.shape
        #   INITIALIZE VIDEO WRITER
        #       Use 'mp4v' for .mp4 output files
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 5.0  # Output video playback speed in FPS
        video_writer = cv2.VideoWriter(
            vid_file_path, fourcc, fps, (width, height)
        )

        # Create our tracker with the first frame and initial OBB.
        mytracker = tracker(first_frame, bbox_init)


        # Initialize file and write header before processing frames
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame"])

            # Loop through the remaining frames in the sequence
            for frame_idx, frame_path in enumerate(frame_paths[1:], start=1):
                frame = cv2.imread(frame_path)

                if frame is None:
                    continue
                
                # Update tracker position for the current frame
                # bbox is (cx, cy, width, height, theta), with theta in radians.
                success, bbox = mytracker.update(frame)

                if success:
                    # Preserve floating-point geometry until rounding for drawing.
                    corners = xywhr2xyxyxyxy(np.array([bbox], dtype=np.float32))[0]
                    draw_corners = np.rint(corners).astype(np.int32)
                    cv2.polylines(frame, [draw_corners], True, (0, 255, 0), 2)

                else:
                    # Render error message if tracking loses target
                    cv2.putText(
                        frame,
                        "Tracking failure",
                        (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                    )

                # Draw the ground-truth box that belongs to this frame, when available.
                if frame_idx < len(gt_lines):
                    x_gt, y_gt, w_gt, h_gt = gt_lines[frame_idx]
                    top_left = (round(x_gt), round(y_gt))
                    bottom_right = (round(x_gt + w_gt), round(y_gt + h_gt))
                    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

                # Overlay live latency and current FPS on the frame
                text = f"Frame: {frame_idx}"
                cv2.putText(
                    frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )

                # Display image frame
                cv2.imshow("Tracking", frame)

                # Record each processed frame without calculating an unused box area.
                writer.writerow([frame_idx])
                video_writer.write(frame)

                # Delay 30ms between frames (~33 FPS playback); exit on 'q'
                #  1. it gives OpenCV time to render the updated image onto your screen
                #  2. it pauses execution briefly to check for keyboard input.
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
            video_writer.release()

        cv2.destroyAllWindows()
    raise


if __name__ == "__main__":
    sequence_folders = sorted(
        glob.glob("/Users/piefacemac/Experiment Data/GOT-10k/val/GOT-10k_Val_*")
    )
    tracking(sequence_folders, "/Users/piefacemac/MyProjects/acm_tracker")
