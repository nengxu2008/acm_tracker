import cv2
import numpy as np
import os, glob, csv, time
from ultralytics.utils.ops import xyxyxyxy2xywhr, xywhr2xyxyxyxy
from ablation import center_error, iou
from formatting import *

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
    def __init__(self, first_frame, init_bbox, max_move=100):
        self.first_frame = first_frame
        self.bbox = init_bbox
        self.max_move = max_move

    def update(self, next_frame):
        """Select the highest-confidence OBB from the two search crops."""
        self.rois, transforms = generate_rois(
            next_frame, self.bbox, max_move=self.max_move
        )
        results = detector(self.rois)

        best_conf = -1.0
        best_bbox = None

        # i identifies the crop; j identifies an object within that crop.
        for i, result in enumerate(results):
            if result.obb is None:
                continue

            for j, confidence in enumerate(result.obb.conf):
                confidence = float(confidence)
                if confidence <= best_conf:
                    continue

                crop_corners = result.obb.xyxyxyxy[j].cpu().numpy()
                frame_corners = cv2.transform(
                    crop_corners[None].astype(np.float32),
                    cv2.invertAffineTransform(transforms[i]),
                )[0]
                candidate_bbox = xyxyxyxy2xywhr(frame_corners[None])[0]

                center_change = np.abs(candidate_bbox[:2] - np.asarray(self.bbox[:2]))
                size_ratio = candidate_bbox[2:4] / np.asarray(self.bbox[2:4])
                if np.any(center_change > self.max_move):
                    continue
                if np.any(size_ratio < 0.5) or np.any(size_ratio > 2.0):
                    continue

                best_bbox = candidate_bbox
                best_conf = confidence

        if best_bbox is None:
            self.success = False
            return self.success, self.bbox

        self.bbox = tuple(float(v) for v in best_bbox)
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
        frame_paths = sorted(glob.glob(os.path.join(seq_folder, "img/*.jpg")))

        # OOTB stores four oriented corner points per frame.
        with open(gt_path, "r") as f:
            gt_corners = np.array(
                [
                    tuple(float(v) for v in line.split())
                    for line in f
                    if line.strip()
                ],
                dtype=np.float32,
            ).reshape(-1, 4, 2)

        # Convert only the first annotation into the tracker's five-value OBB state.
        bbox_init = tuple(float(v) for v in xyxyxyxy2xywhr(gt_corners[:1])[0])

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

        latencies_pv = []
        center_error_pv = []
        iou_pv = []
        successful_frames = 0

        # Initialize file and write header before processing frames
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "frame",
                    "success",
                    "latency_ms",
                    "tracker_fps",
                    "oriented_iou",
                    "center_error_px",
                    "cx",
                    "cy",
                    "width",
                    "height",
                    "theta",
                ]
            )

            # Loop through the remaining frames in the sequence
            for frame_idx, frame_path in enumerate(frame_paths[1:], start=1):
                frame = cv2.imread(frame_path)

                if frame is None:
                    continue

                # Update tracker position for the current frame
                # bbox is (cx, cy, width, height, theta), with theta in radians.
                bbox_gt = xyxyxyxy2xywhr(gt_corners[frame_idx : frame_idx + 1])[0]
                start_time = time.perf_counter()
                success, bbox = mytracker.update(frame)
                end_time = time.perf_counter()

                # calculate latency in milliseconds
                latency_ms = (end_time - start_time) * 1000
                latencies_pv.append(latency_ms)

                # Set default values for failed tracking frames
                iou_pf = 0.0
                center_error_pf = float("nan")

                if success:
                    successful_frames += 1

                    # Preserve floating-point geometry until rounding for drawing.
                    corners = xywhr2xyxyxyxy(np.array([bbox], dtype=np.float32))[0]
                    draw_corners = np.rint(corners).astype(np.int32)
                    cv2.polylines(frame, [draw_corners], True, (0, 255, 0), 2)

                    # calculate iou and center_error
                    center_error_pf = center_error(bbox, bbox_gt)
                    iou_pf = iou(bbox, bbox_gt)

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

                # Failed frames contribute zero overlap and undefined center error.
                iou_pv.append(iou_pf)
                center_error_pv.append(center_error_pf)

                # Draw the ground-truth box that belongs to this frame, when available.
                if frame_idx < len(gt_corners):
                    draw_gt_corners = np.rint(gt_corners[frame_idx]).astype(np.int32)
                    cv2.polylines(frame, [draw_gt_corners], True, (0, 0, 255), 2)

                # Overlay live latency and current FPS on the frame
                fps = 1000 / latency_ms if latency_ms > 0 else 0
                center_error_text = (
                    f"{center_error_pf:.1f}px"
                    if np.isfinite(center_error_pf)
                    else "N/A"
                )

                draw_stats(frame, frame_idx, fps, iou_pf, center_error_text)

                # Display image frame
                cv2.imshow("Tracking", frame)

                # Record the same per-frame statistics shown on the video.
                bbox_values = (
                    [float(value) for value in bbox]
                    if success
                    else [""] * 5
                )
                writer.writerow(
                    [
                        frame_idx,
                        success,
                        latency_ms,
                        fps,
                        iou_pf,
                        "" if not np.isfinite(center_error_pf) else center_error_pf,
                        *bbox_values,
                    ]
                )
                video_writer.write(frame)

                # Delay 30ms between frames (~33 FPS playback); exit on 'q'
                #  1. it gives OpenCV time to render the updated image onto your screen
                #  2. it pauses execution briefly to check for keyboard input.
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
            video_writer.release()

        cv2.destroyAllWindows()

        # Compute overall video stats
        if not latencies_pv:
            return None

        processed_frames = len(latencies_pv)
        avg_latency = float(np.mean(latencies_pv))
        avg_fps = 1000 / avg_latency if avg_latency > 0 else 0.0
        avg_iou = float(np.mean(iou_pv))
        tracked_rate = successful_frames / processed_frames
        valid_center_errors = [
            value for value in center_error_pv if np.isfinite(value)
        ]
        avg_center_error = (
            float(np.mean(valid_center_errors))
            if valid_center_errors
            else float("nan")
        )

        summary = {
            "sequence": folder_name,
            "processed_frames": processed_frames,
            "tracked_rate": tracked_rate,
            "avg_latency_ms": avg_latency,
            "avg_tracker_fps": avg_fps,
            "avg_oriented_iou": avg_iou,
            "avg_center_error_tracked_px": avg_center_error,
        }
        print("Tracking summary:", summary)

        # This experiment intentionally runs only the first selected sequence.
        return summary

if __name__ == "__main__":
    sequence_folders = sorted(
        glob.glob("/Users/piefacemac/Experiment Data/OOTB dataset/ship_*")
    )
    tracking(sequence_folders, "/Users/piefacemac/MyProjects/acm_tracker")
