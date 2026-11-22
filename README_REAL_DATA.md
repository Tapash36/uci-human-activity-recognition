# Recording & Evaluating Your Own Smartphone Data (`evaluate_real_data.py`)

This guide explains how to record 50 Hz accelerometer and gyroscope data from your own smartphone, format the CSV file, and run `evaluate_real_data.py` to get activity predictions and accuracy metrics.

---

## 1. How to Record 50 Hz Phone Data

1. **Mount Position:** Place your smartphone securely at your **waist** (belt clip, waistband, or front waist pocket), matching the UCI HAR protocol.
2. **Recommended Free Sensor Apps (iOS / Android):**
   - **Sensor Logger** (Android / iOS)
   - **Physics Toolbox Sensor Suite** (Android / iOS)
   - **phyphox** (Android / iOS)
3. **Recording Settings:**
   - Enable **Accelerometer** (Total Acceleration including gravity) and **Gyroscope** (Angular Velocity).
   - Set the sampling rate to **50 Hz** (`20 ms` interval).
   - Export the recording as a **CSV** file.

---

## 2. Required CSV Format

Your CSV file must have **at least 128 rows** (`2.56 seconds` at 50 Hz) and contain 3-axis accelerometer and 3-axis gyroscope columns:

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
0.00,0.892,-0.231,0.114,0.012,-0.008,0.004
0.02,0.895,-0.228,0.116,0.015,-0.006,0.003
...
```

### Supported Column Names
- **Accelerometer:** `acc_x, acc_y, acc_z` (or `ax, ay, az` or `accelerometer_x, accelerometer_y, accelerometer_z`)
- **Gyroscope:** `gyro_x, gyro_y, gyro_z` (or `gx, gy, gz` or `gyroscope_x, gyroscope_y, gyroscope_z`)
- **Units:**
  - Accelerometer can be in **$g$** (`1g = 9.80665 m/s²`) or **$\text{m/s}^2$** (`--units auto` automatically detects $\text{m/s}^2$ and divides by `9.80665`).
  - Gyroscope defaults to **$\text{rad/s}$** (`--gyro-units rad`). If your app exports degrees/sec, pass `--gyro-units deg`.

### Optional Ground-Truth Labels
To compute accuracy and a classification report, you can either:
1. Include an `activity` (or `label`) column directly in your CSV using either activity names (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) or IDs (`1` to `6`), **OR**
2. Pass a separate labels file via `--labels my_labels.csv` (containing one label per row or one label per 128-sample window).

---

## 3. What `evaluate_real_data.py` Does Internally

1. **Median & Low-Pass Noise Filtering:** Applies a 3-point median filter and a 3rd-order low-pass Butterworth filter (`20 Hz` cutoff at `50 Hz` sampling rate) to remove high-frequency sensor noise.
2. **Gravity / Body Separation:** Applies a 3rd-order low-pass Butterworth filter (`0.3 Hz` cutoff) to isolate gravitational acceleration (`total_acc` $\rightarrow$ `gravity_acc`) and subtracts it to obtain `body_acc = total_acc - gravity_acc`.
3. **128×9 Sliding Windowing:** Segments the 9 channels (`total_acc_x/y/z`, `body_acc_x/y/z`, `body_gyro_x/y/z`) into windows of `128` samples (`2.56 s`) with `50%` overlap (`step = 64` samples).
4. **Model Inference:** Standardizes the windows using the training set statistics and runs the trained PyTorch `HAR_1DCNN` model to output per-window predictions, confidence scores, activity distribution, and (if labels are provided) overall accuracy and per-class precision/recall/F1.

---

## 4. Example Commands

```bash
# Predict activities on an unlabeled phone recording:
python evaluate_real_data.py --csv my_phone_recording.csv

# Predict and evaluate accuracy when the CSV includes an 'activity' column:
python evaluate_real_data.py --csv my_labeled_recording.csv

# Predict and evaluate accuracy with a separate labels file and deg/s gyroscope units:
python evaluate_real_data.py --csv my_phone_recording.csv --labels my_labels.csv --gyro-units deg
```
