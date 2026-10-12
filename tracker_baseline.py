from detector import model as detector
import csv
import cv2
import os


def tracking(sequence_folders, output_dir):
    """Run YOLO's official ByteTrack mechanism on OOTB image sequences."""
    os.makedirs(output_dir, exist_ok=True)

    for seq_folder in sequence_folders:
        folder_name = os.path.basename(seq_folder)
        image_folder = os.path.join(seq_folder, "img")
        csv_path = os.path.join(
            output_dir,
            f"baseline_bytetrack_{folder_name}.csv",
        )
        video_path = os.path.join(
            output_dir,
            f"baseline_bytetrack_{folder_name}.mp4",
        )

        # Start ByteTrack with fresh state for this independent sequence.
        detector.predictor = None

        # The OBB detector still produces the oriented boxes; ByteTrack adds IDs.
        results = detector.track(
            source=image_folder,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.1,
            show=False,
            save=False,
            stream=True,
            verbose=False,
        )

        video_writer = None
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "frame",
                    "track_id",
                    "class_id",
                    "confidence",
                    "cx",
                    "cy",
                    "width",
                    "height",
                    "theta",
                ]
            )

            for frame_idx, result in enumerate(results):
                annotated_frame = result.plot()
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        5.0,
                        (width, height),
                    )
                video_writer.write(annotated_frame)

                if result.obb is None or result.obb.id is None:
                    writer.writerow([frame_idx, "", "", "", "", "", "", "", ""])
                else:
                    boxes = result.obb.xywhr.cpu().numpy()
                    track_ids = result.obb.id.int().cpu().numpy()
                    class_ids = result.obb.cls.int().cpu().numpy()
                    confidences = result.obb.conf.cpu().numpy()

                    for bbox, track_id, class_id, confidence in zip(
                        boxes,
                        track_ids,
                        class_ids,
                        confidences,
                    ):
                        writer.writerow(
                            [
                                frame_idx,
                                int(track_id),
                                int(class_id),
                                float(confidence),
                                *[float(value) for value in bbox],
                            ]
                        )

                cv2.imshow("YOLO OBB + ByteTrack baseline", annotated_frame)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break

        results.close()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        return


if __name__ == "__main__":
    sequence_folders = [
        "/Users/piefacemac/Experiment Data/OOTB dataset/ship_1"
    ]
    tracking(sequence_folders, "/Users/piefacemac/MyProjects/acm_tracker")
