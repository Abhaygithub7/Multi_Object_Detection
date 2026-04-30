# Technical Report: Multi-Object Detection and Persistent ID Tracking

**Author:** Abhay Singh Chauhan  
**Date:** April 2026  

---

## 1. Model / Detector Used

**YOLOv8m (Medium)** from the Ultralytics library was selected as the object detector. YOLOv8 represents the latest generation of the YOLO family, offering an excellent balance between inference speed and detection accuracy. The medium variant (`yolov8m.pt`) was chosen as a deliberate trade-off — larger than the small/nano variants for better accuracy on distant or partially-occluded players, while remaining fast enough for offline video processing at 25 fps.

The model was used with its pre-trained COCO weights without any fine-tuning. COCO class `0` (person) was the sole detection target, which proved sufficient for detecting players, referees, and other personnel in the football footage. A confidence threshold of `0.25` was applied at the detector level.

## 2. Tracking Algorithm Used

**ByteTrack** was selected as the multi-object tracking algorithm, accessed through Roboflow's `supervision` library. ByteTrack is distinguished by its two-stage association approach:

1. **First pass**: High-confidence detections (above `track_activation_threshold`) are matched to existing tracks using IoU and a Kalman filter for motion prediction.
2. **Second pass**: Low-confidence detections (between 0.1 and the threshold) are matched to any remaining unmatched tracks, allowing partially-occluded subjects to maintain their IDs.

This dual-pass strategy is critical for sports footage where players frequently overlap and partially occlude each other, generating low-confidence detections that would be discarded by single-pass trackers like vanilla SORT.

## 3. Why This Combination Was Selected

| Criterion | YOLOv8 + ByteTrack |
|-----------|---------------------|
| **Real-time capability** | YOLOv8 processes frames in ~20ms; ByteTrack adds negligible overhead |
| **Occlusion robustness** | ByteTrack's two-stage matching utilises weak detections from occluded players |
| **No ReID dependency** | ByteTrack uses IoU + Kalman prediction, avoiding the computational cost and complexity of ReID feature extraction |
| **Library maturity** | Both `ultralytics` and `supervision` are actively maintained with clean APIs |
| **Proven track record** | ByteTrack achieved state-of-the-art results on MOT17/MOT20 benchmarks |

An alternative considered was **BoT-SORT**, which adds camera motion compensation (CMC) and a ReID module. While BoT-SORT would further improve tracking in extreme camera pan/zoom scenarios, ByteTrack's simpler architecture proved sufficient for the selected footage and kept the pipeline's complexity and dependencies manageable.

## 4. How ID Consistency Is Maintained

ID consistency is achieved through a combination of algorithmic and parameter-tuning strategies:

### Algorithmic (ByteTrack internals)
- **Kalman Filter prediction**: Each track's position is predicted forward in time before association, accounting for linear motion between frames.
- **Two-stage IoU matching**: Weak detections from partial occlusions still match existing tracks, preventing premature ID loss.
- **Track lifecycle management**: Tracks transition through states (New → Confirmed → Lost → Removed) with configurable thresholds.

### Parameter Tuning (our config)
Four key ByteTrack parameters were tuned for sports footage:

| Parameter | Default → Tuned | Effect |
|-----------|-----------------|--------|
| `track_activation_threshold` | 0.25 → **0.20** | Seeds tracks from weaker detections (distant/occluded players) |
| `lost_track_buffer` | 30 → **45** | Retains lost IDs for ~1.5–1.8s before deletion |
| `minimum_matching_threshold` | 0.8 → **0.70** | Permits matching across larger inter-frame displacements |
| `minimum_consecutive_frames` | 1 → **3** | Suppresses spurious one-frame false positives |

These tuning decisions were validated through a comprehensive TDD test suite (13 tracker-specific tests) that simulates occlusion scenarios, rapid movement, and low-confidence detection patterns.

## 5. Challenges Faced

1. **Supervision API changes**: The `supervision` library's annotator class names changed between versions (`BoundingBoxAnnotator` → `BoxAnnotator`), and the `TraceAnnotator` requires `ColorLookup.TRACK` when detections lack `class_id`. These were resolved by reading the supervision source code directly.

2. **Empty detection handling**: Passing empty detection arrays to `sv.Detections()` required a 2D array with shape `(0, 4)`, not a 1D empty array. The fix was to use `sv.Detections.empty()`.

3. **ByteTrack threshold semantics**: The `track_activation_threshold` interacts with an internal `det_thresh = threshold + 0.1`, meaning detections must exceed 0.30 (not 0.20) to seed *new* tracks. Low-confidence detections only maintain *existing* tracks through second-pass matching. Understanding this required source-code inspection.

4. **Team clustering initialisation**: K-Means requires a calibration phase — we collect detection crops from the first 30 frames before fitting the model. This introduces a brief delay before team colours are assigned, during which boxes use default colours.

## 6. Failure Cases Observed

- **Close-proximity ID switches**: When two players cross paths at close range (within ~30px), their bounding boxes overlap significantly, occasionally causing the IoU-based matcher to swap IDs.
- **Referee misclassification**: The referee's kit colour may be clustered as a "third team" or incorrectly merged with one of the two teams, since K-Means is configured for `n_teams=2`.
- **Scale-dependent detection**: Players far from the camera (near the top of the frame) produce small bounding boxes that YOLOv8m occasionally misses, especially when partially occluded by closer players.
- **Broadcast graphics overlap**: Any on-screen graphics (score overlays, watermarks) that overlap with player regions can confuse both detection and team-colour clustering.

## 7. Possible Improvements

1. **BoT-SORT integration**: Adding camera motion compensation (GMC) and a ReID module (e.g., OSNet) would significantly reduce ID switches during fast camera pans and long-term occlusions.

2. **Higher-resolution input**: Processing at 1080p instead of 360p would improve detection of distant players, at the cost of ~4× slower inference.

3. **Sport-specific fine-tuning**: Fine-tuning YOLOv8 on a football-specific dataset (e.g., SoccerNet) would improve detection of players in challenging poses and improve the detection of sport-specific objects (ball, goalposts).

4. **Improved team classification**: Replacing K-Means with a more robust method such as:
   - Gaussian Mixture Models (GMM) for soft clustering
   - Extracting features from the *torso region only* (excluding shorts/socks) for cleaner jersey-colour isolation
   - Adding a third cluster for referees and filtering them out

5. **Bird's-eye view projection**: Using homography estimation to project player positions onto a top-down pitch view would enable speed estimation, formation analysis, and heatmap generation.

6. **Evaluation metrics**: Implementing standard MOT metrics (MOTA, IDF1, HOTA) using the `py-motmetrics` library would enable quantitative comparison of tracker configurations.

---

*Total word count: ~950 words (within the 1–2 page requirement)*
