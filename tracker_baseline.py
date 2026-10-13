from detector import model as detector
import csv
import cv2
import os, glob, time
import numpy as np
from ablation import center_error, iou
from formatting import *
from ultralytics.utils.ops import xyxyxyxy2xywhr, xywhr2xyxyxyxy


class bytetrack_baseline:
    """YOLO OBB + ByteTrack wrapped as a single-object tracker.
    
        Same logic as in the mytracker.py
    """

    def __init__(self, first_frame, init_bbox, conf=0.1, min_init_iou=0.1):
        self.bbox = init_bbox
        self.conf = conf
        self.min_init_iou = min_init_iou
        self.target_id = None

        # Fresh ByteTrack state for this independent sequence.
        detector.predictor = None

        # Frame 0 initializes ByteTrack
        boxes, ids = self._track(first_frame)
        self._initialize_target(boxes, ids)

        if self.target_id is None:
            print(
                "ByteTrack has not confirmed the initial target yet; "
                "target locking will continue on later frames."
            )

    def _track(self, frame):
        """Run detector + ByteTrack on one frame; return (xywhr boxes, track ids)."""
        result = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            verbose=False,
        )[0]

        obb = result.obb
        if obb is None:
            return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=int)

        boxes = obb.xywhr.cpu().numpy()
        ids = (
            obb.id.int().cpu().numpy()
            if obb.id is not None
            else np.full(len(boxes), -1, dtype=int)
        )
        return boxes, ids

    def _initialize_target(self, boxes, ids):
        """Lock onto the confirmed track that overlaps the initial target box."""
        confirmed = np.flatnonzero(ids >= 0)
        if len(confirmed) == 0:
            return False

        overlaps = np.array([iou(boxes[index], self.bbox) for index in confirmed])
        best = int(np.argmax(overlaps))
        if overlaps[best] < self.min_init_iou:
            return False

        box_index = int(confirmed[best])
        self.target_id = int(ids[box_index])
        self.bbox = tuple(float(value) for value in boxes[box_index])
        return True

    def update(self, next_frame):
        """Return (success, bbox) for the locked-on track ID."""
        boxes, ids = self._track(next_frame)

        if self.target_id is None:
            self.success = self._initialize_target(boxes, ids)
            return self.success, self.bbox

        matches = np.flatnonzero(ids == self.target_id)
        if len(matches) == 0:
            # Target ID not present on this frame: count as a failure.
            self.success = False
            return self.success, self.bbox

        self.bbox = tuple(float(v) for v in boxes[matches[0]])
        self.success = True
        return self.success, self.bbox



def tracking(sequence_folders, output_dir):
    """Run YOLO's official ByteTrack mechanism on OOTB image sequences."""
    os.makedirs(output_dir, exist_ok=True)

    for seq_folder in sequence_folders:
        if not os.path.isdir(seq_folder):
            continue

        # Create file name
        folder_name = os.path.basename(seq_folder)
        csv_filename = f"baseline_bytetrack_{folder_name}.csv"
        csv_file_path = os.path.join(output_dir, csv_filename)
        vid_filename = f"baseline_bytetrack_{folder_name}.mp4"
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

        # Read the first frame to initialize the tracker
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

        # Create the baseline tracker with the first frame and initial OBB.
        mytracker = bytetrack_baseline(first_frame, bbox_init)
      

        latencies_pv = []
        center_error_pv = []
        iou_pv = []
        successful_frames = 0

        with open(csv_file_path, "w", newline="") as file:
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

            for frame_idx, frame_path in enumerate(frame_paths[1:], start=1):
                # Frame 0 initializes the target sequence; evaluate from frame 1,
                # matching tracker.py.
                frame = cv2.imread(frame_path)

                if frame is None:
                    continue

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

                    # Calculate iou and center_error
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
                cv2.imshow("YOLO OBB + ByteTrack baseline", frame)

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

        return summary




if __name__ == "__main__":
    sequence_folders = [
        "/Users/piefacemac/Experiment Data/OOTB dataset/ship_1"
    ]
    tracking(sequence_folders, "/Users/piefacemac/MyProjects/acm_tracker")
