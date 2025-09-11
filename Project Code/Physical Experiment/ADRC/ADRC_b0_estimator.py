import os
import cv2
import multiprocessing
import pupil_apriltags as apriltag
import pyaudio
import audioop
import wave
import numpy as np
import math
import time
import scipy.signal as signal
from Siyuan_NNF_0703 import my_NNF
from FV_NNF import Fz_deltaV_NNF

"""
Purpose
-------
This is the **b0 estimation mode** (no ADRC, no Smith predictor). We drive the robot
by sending the **reference trajectory directly into `my_NNF`** (optionally add Fz
compensation) and estimate per‑axis b0 from measured motion.

Compared to the first version, this script adds **robust b0 estimation** tailored for
0.4 mm step tests (default), using:
  • Center‑difference derivative  \dot y_k ≈ (y_{k+1}-y_{k-1})/(T_{k}+T_{k-1})
  • Delay‑aligned (we pair r and y at the same timestamp, open‑loop)
  • Error‑window filtering: keep |e| in [0.05A, 0.8A]
  • Huber‑weighted least squares to suppress outliers
  • Report standard error and sample count

Outputs are saved to Output_b0_est/:
  - b0_estimates.txt  (with ±95% CI and N)
  - b0_data.csv       (t, r_x, y_x, r_y, y_y, r_z, y_z)
  - detected_path.csv ([x, -y, z, t] for quick plotting)
  - control_co_*.csv  (normalized voltages streamed)

NOTE on units: r and y must be in the **same unit** (mm or m). This script assumes
your desired_path.csv matches the unit of the learned mapping and your measured
pose is in the same unit after any project‑specific scaling. The Y sign follows your
original pipeline: we negate desired Y before NNF, measured Y stays positive; the
regression uses that same (r_used, y_meas) pair.
"""

# =============================================================================================
# Config
# =============================================================================================
folder_name = "Output_b0_est"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Step amplitude used for robust windowing (match your test). Default: 0.4 (e.g., 0.4 mm)
A_STEP = 0.4
E_LOW_FACTOR  = 0.05   # keep |e| >= 0.05A
E_HIGH_FACTOR = 0.80   # keep |e| <= 0.80A

# =============================================================================================
# Utilities
# =============================================================================================
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.filtered_value = None
    def update(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

# Robust WLS for dydt = b0 * e
def robust_b0_from_series(e_series, dydt_series):
    e = np.asarray(e_series, float)
    dydt = np.asarray(dydt_series, float)
    if e.size < 20:
        return float('nan'), float('nan'), 0

    # 1) Error windowing
    e_min = E_LOW_FACTOR  * A_STEP
    e_max = E_HIGH_FACTOR * A_STEP
    m = (np.abs(e) >= e_min) & (np.abs(e) <= e_max)
    e = e[m]; dydt = dydt[m]
    if e.size < 20:
        return float('nan'), float('nan'), int(e.size)

    # 2) Huber weights on residuals (initial slope from median ratio)
    r_ratio = dydt / (e + 1e-12)
    b0_0 = np.median(r_ratio)
    resid0 = dydt - b0_0 * e
    mad = np.median(np.abs(resid0 - np.median(resid0))) + 1e-12
    delta = 1.345 * mad
    w = np.ones_like(resid0)
    idx = np.abs(resid0) > delta
    w[idx] = delta / (np.abs(resid0[idx]) + 1e-12)

    # 3) Weighted LS
    num = np.sum(w * dydt * e)
    den = np.sum(w * e * e) + 1e-12
    b0_hat = num / den

    # 4) SE
    resid = dydt - b0_hat * e
    dof = max(1, e.size - 1)
    sigma2 = np.sum(w * resid**2) / dof
    se = math.sqrt(sigma2 / den)
    return float(b0_hat), float(se), int(e.size)

# =============================================================================================
# Video capture process
# =============================================================================================

def video_capture(queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    while trigger.value:
        try:
            ret, frame = cap.read()
        except Exception:
            continue
        if not ret:
            print("Failed to read frame from camera")
            break
        queue.put(frame)
    print('Video end')
    cap.release()

# =============================================================================================
# Image processing + open‑loop NNF control + robust b0 estimation
# =============================================================================================

def image_processing(queue):
    global desired_trajectory, num_cols, n

    at_detector = apriltag.Detector(families='tag36h11')
    blank_layer = np.zeros((480, 640, 3))

    # Log measured path for visualization (x, -y, z, t)
    path1 = np.array([[0], [0], [0], [0]])

    # Mild smoothing on vision only
    alpha = 0.2
    lpf_x = LowPassFilter(alpha)
    lpf_y = LowPassFilter(alpha)
    lpf_z = LowPassFilter(alpha)

    t0 = time.time()
    tag_origin = np.array([0, 0, 0])

    # ====== Buffers to build center‑difference derivatives ======
    # We store full sequences, then compute dydt via center differences.
    r_hist = {'x': [], 'y': [], 'z': []}
    y_hist = {'x': [], 'y': [], 'z': []}
    Ts_hist = []

    # Raw log for offline inspection
    b0_log_rows = []  # [t, r_x, y_x, r_y, y_y, r_z, y_z]

    # Timing
    t_last = time.perf_counter()

    # Origin averaging startup
    f = True
    n_origin = 0

    while True:
        if queue.empty():
            continue
        frame = queue.get()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            gray_frame,
            estimate_tag_pose=True,
            camera_params=[692.299318794764, 692.371474887306, 310.832011088224, 213.144290469954],
            tag_size=3,
        )

        # Real sampling period (vision loop)
        t_now = time.perf_counter()
        Ts_real = t_now - t_last
        t_last = t_now
        Ts_real = max(1e-4, min(0.1, Ts_real))

        # Initial origin acquisition
        if f:
            if len(tags) > 0 and n_origin < 20:
                tag_origin = tags[0].pose_t
                n_origin += 1
                continue
            elif n_origin > 100:
                f = False
                continue
            elif n_origin == 20:
                print(tag_origin)
                n_origin += 1
                continue
            else:
                n_origin += 1
                continue

        if len(tags) > 0 and tags[0].tag_id == 0:
            # Measured pose (meters, or your working unit)
            tag_p = tags[0].pose_t - tag_origin

            # Log path (x, -y, z, t)
            t_abs = time.time() - t0
            path1 = np.concatenate(
                (path1, np.array([[tag_p[0][0]], [-tag_p[1][0]], [tag_p[2][0]], [t_abs]])), axis=1
            )

            # Grab current desired sample
            n_num = n.value
            if n_num < num_cols:
                # desired_trajectory rows: [x; y; z; Fz]
                r_x = desired_trajectory[0, n_num]
                r_y = -desired_trajectory[1, n_num]  # sign to match original pipeline
                r_z = desired_trajectory[2, n_num]
                desired_Fz = desired_trajectory[3, n_num]
            else:
                break

            # LPF measured pose (vision noise only)
            yx = lpf_x.update(tag_p[0][0])
            yy = lpf_y.update(tag_p[1][0])  # positive
            yz = lpf_z.update(tag_p[2][0])

            # --- save sequences for robust estimation ---
            r_hist['x'].append(r_x); y_hist['x'].append(yx)
            r_hist['y'].append(r_y); y_hist['y'].append(yy)
            r_hist['z'].append(r_z); y_hist['z'].append(yz)
            Ts_hist.append(Ts_real)

            # Raw log row for inspection
            b0_log_rows.append([t_abs, r_x, yx, r_y, yy, r_z, yz])

            # ========================
            # Open‑loop control via NNF
            # ========================
            nnf_input = np.array([[r_x], [r_y], [r_z]], dtype=float)

            if abs(desired_Fz) < 1e-6:
                v_cmd = my_NNF(nnf_input)
            else:
                input_with_fz = np.vstack([nnf_input, [[desired_Fz]]])
                v_cmd = my_NNF(nnf_input) + Fz_deltaV_NNF(input_with_fz)

            # Clip to [-1, 1] for the audio path
            control_1.value = float(np.clip(v_cmd[0][0], -1.0, 1.0))
            control_2.value = float(np.clip(v_cmd[1][0], -1.0, 1.0))
            control_3.value = float(np.clip(v_cmd[2][0], -1.0, 1.0))

            # Overlay
            cnd = blank_layer[:, :, :] > 0
            frame[cnd] = blank_layer[cnd]

        cv2.imshow('Processed Video (b0 estimation mode: robust 0.4mm)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop other loops
    trigger.value = 0
    cv2.destroyAllWindows()

    # ========================
    # Save raw logs
    # ========================
    file_path = os.path.join(folder_name, "detected_path.csv")
    np.savetxt(file_path, path1, delimiter=",")

    file_path = os.path.join(folder_name, "b0_data.csv")
    np.savetxt(file_path, np.asarray(b0_log_rows), delimiter=",")

    # ========================
    # Robust estimation (center differences + window + Huber WLS)
    # ========================
    def center_diff(y_list, Ts_list):
        y = np.asarray(y_list, float)
        Ts = np.asarray(Ts_list, float)
        if y.size < 3:
            return np.array([])
        dydt = np.zeros_like(y)
        # interior points
        for k in range(1, y.size-1):
            Tk = Ts[k] + Ts[k-1]
            dydt[k] = (y[k+1] - y[k-1]) / max(Tk, 1e-9)
        # drop the two edges for regression
        return dydt[1:-1], y[1:-1]

    # Build series per axis
    results = {}
    for ax in ['x','y','z']:
        dydt, y_mid = center_diff(y_hist[ax], Ts_hist)
        if dydt.size == 0:
            results[ax] = (float('nan'), float('nan'), 0)
            continue
        r_mid = np.asarray(r_hist[ax], float)[1:-1]
        e_mid = r_mid - y_mid
        b0_hat, se, N = robust_b0_from_series(e_mid, dydt)
        results[ax] = (b0_hat, se, N)

    # Print & save
    lines = [
        "==== b0 estimates (robust; dydt = b0 * e; A=%.3f) ====\n" % A_STEP,
    ]
    axis_names = {'x':'b0_x_hat','y':'b0_y_hat','z':'b0_z_hat'}
    for ax in ['x','y','z']:
        b0_hat, se, N = results[ax]
        if not np.isfinite(b0_hat):
            lines.append(f"{axis_names[ax]} = NaN  [1/s]  (N={N})")
        else:
            ci95 = 1.96 * se
            lines.append(f"{axis_names[ax]} = {b0_hat:.6f}  [1/s]  ± {ci95:.6f}  (SE={se:.6f}, N={N})")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(folder_name, "b0_estimates.txt"), 'w') as f:
        f.write(txt + "\n")

# =============================================================================================
# Audio path + main
# =============================================================================================

def butter_lowpass_filter(data, zi):
    global b, a
    y, zi = signal.lfilter(b, a, data, zi=zi)
    return y, zi

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    control_1 = multiprocessing.Value('d', 0)
    control_2 = multiprocessing.Value('d', 0)
    control_3 = multiprocessing.Value('d', 0)
    trigger = multiprocessing.Value('i', 1)

    print("Camera Created")
    audio = pyaudio.PyAudio()

    # === Adjust folder to your trajectory files ===
    folder = "/home/cyrus/MScProject/ProjectCode/Trajectory/z_direction/"  # use your 0.4 mm step set

    wav_file1_name = folder + "control01.wav"
    wav_file1 = wave.open(wav_file1_name, 'rb')
    wav_file2_name = folder + "control01.wav"
    wav_file2 = wave.open(wav_file2_name, 'rb')
    wav_file3_name = folder + "control01.wav"
    wav_file3 = wave.open(wav_file3_name, 'rb')

    print("Voltage Data Imported")

    # === Adjust device indices to match your setup ===
    device_index1 = 3
    device_info1 = audio.get_device_info_by_index(device_index1)
    device_index2 = 2
    device_info2 = audio.get_device_info_by_index(device_index2)

    print(f"Playing audio on {device_info1['name']} and {device_info2['name']}")

    stream1 = audio.open(format=audio.get_format_from_width(wav_file1.getsampwidth()),
                         channels=1,
                         rate=wav_file1.getframerate(),
                         output=True,
                         output_device_index=device_index1)
    stream2 = audio.open(format=audio.get_format_from_width(wav_file2.getsampwidth()),
                         channels=2,
                         rate=wav_file2.getframerate(),
                         output=True,
                         output_device_index=device_index2)
    print("Player Created")

    samples = 2

    global order, zi, b, a, duration, num_channels
    order = 2
    cutoff_freq = 2
    sampling_freq = 1000
    normalized_cutoff_freq = 2.0 * cutoff_freq / sampling_freq
    b, a = signal.butter(order, normalized_cutoff_freq, btype='lowpass', analog=False)

    zi1 = signal.lfiltic(b, a, 0)
    zi2 = signal.lfiltic(b, a, 0)
    zi3 = signal.lfiltic(b, a, 0)

    # Audio‑side gentle smoothing of voltages
    alpha = 0.2
    low_pass_filter1 = LowPassFilter(alpha)
    low_pass_filter2 = LowPassFilter(alpha)
    low_pass_filter3 = LowPassFilter(alpha)

    print("Filter Created")

    # ===== Load desired trajectory (0.4 mm step sequence) =====
    trajectory_file_name = folder + "desired_path.csv"
    desired_trajectory = np.genfromtxt(trajectory_file_name, delimiter=',')
    num_rows, num_cols = desired_trajectory.shape
    n = multiprocessing.Value('i', 0)

    WIDTH1 = wav_file1.getsampwidth()
    WIDTH2 = wav_file2.getsampwidth()
    WIDTH3 = wav_file3.getsampwidth()
    WIDTH = 2

    data1 = wav_file1.readframes(samples)
    data2 = wav_file2.readframes(samples)
    data3 = wav_file3.readframes(samples)

    # Initial scaling factors (kept as 1, voltage magnitude comes from controls)
    FACTOR1 = 1
    FACTOR2 = 1
    FACTOR3 = 1

    # Start camera and processing
    video_process = multiprocessing.Process(target=video_capture, args=(queue,))
    video_process.start()
    print("Video Started")

    image_process = multiprocessing.Process(target=image_processing, args=(queue,))
    image_process.start()
    print("Image Processing Started")

    time.sleep(10)
    n_x = 0

    # Buffers for audio stream blending
    data_1_co = [0]
    data_2_co = [0]
    data_3_co = [0]
    data_1_co_former = [0]
    data_2_co_former = [0]
    data_3_co_former = [0]
    control_co_1 = [0]
    control_co_2 = [0]
    control_co_3 = [0]

    # start with small non‑zero to satisfy PyAudio write
    data1 = b'\xff\x7f' * samples
    data2 = b'\xff\x7f' * samples
    data3 = b'\xff\x7f' * samples
    data_row = b'\xff\x7f'

    data1 = audioop.mul(data1, WIDTH1, data_1_co[0])
    data2 = audioop.mul(data2, WIDTH2, data_2_co[0])
    data3 = audioop.mul(data3, WIDTH3, data_3_co[0])

    data_2 = np.frombuffer(data2, dtype=np.int16).reshape(-1, 1)
    data_3 = np.frombuffer(data3, dtype=np.int16).reshape(-1, 1)
    data_4 = np.concatenate((data_2, data_3), axis=1)

    data2_2 = data_4.tobytes()
    data1_2 = data1
    print("Voltage Supply Started")

    # ===== Main audio streaming loop =====
    while n.value < num_cols:
        data_1_co_former = data_1_co
        data_2_co_former = data_2_co
        data_3_co_former = data_3_co

        # read shared controls (normalized)
        with control_1.get_lock():
            v1_val = control_1.value
        with control_2.get_lock():
            v2_val = control_2.value
        with control_3.get_lock():
            v3_val = control_3.value

        # Audio‑side LPF
        data_1_co, zi1 = butter_lowpass_filter(np.array([v1_val]), zi1)
        data_2_co, zi2 = butter_lowpass_filter(np.array([v2_val]), zi2)
        data_3_co, zi3 = butter_lowpass_filter(np.array([v3_val]), zi3)

        data_1_co[0] = low_pass_filter1.update(data_1_co[0])
        data_2_co[0] = low_pass_filter2.update(data_2_co[0])
        data_3_co[0] = low_pass_filter3.update(data_3_co[0])

        stream1.write(data1_2, samples, exception_on_underflow=True)
        stream2.write(data2_2, samples, exception_on_underflow=True)

        n_x += 1
        if n_x >= 2:
            data1 = audioop.mul(data_row, WIDTH1, data_1_co_former[0]) + audioop.mul(data_row, WIDTH1, data_1_co[0])
            data2 = audioop.mul(data_row, WIDTH2, data_2_co_former[0]) + audioop.mul(data_row, WIDTH2, data_2_co[0])
            data3 = audioop.mul(data_row, WIDTH3, data_3_co_former[0]) + audioop.mul(data_row, WIDTH3, data_3_co[0])

            control_co_1.append(data_1_co[0])
            control_co_2.append(data_2_co[0])
            control_co_3.append(data_3_co[0])

            data_2 = np.frombuffer(data2, dtype=np.int16).reshape(-1, 1)
            data_3 = np.frombuffer(data3, dtype=np.int16).reshape(-1, 1)
            data_4 = np.concatenate((data_2, data_3), axis=1)
            data2_2 = data_4.tobytes()
            data1_2 = data1

            n.value += 2
            n_x = 0

    print("Playing stops")

    # Save streamed controls
    file_path = os.path.join(folder_name, "control_co_1.csv")
    np.savetxt(file_path, control_co_1, delimiter=",")
    file_path = os.path.join(folder_name, "control_co_2.csv")
    np.savetxt(file_path, control_co_2, delimiter=",")
    file_path = os.path.join(folder_name, "control_co_3.csv")
    np.savetxt(file_path, control_co_3, delimiter=",")

    # Cleanup
    stream1.close()
    stream2.close()
    wav_file1.close()
    wav_file2.close()
    wav_file3.close()
    desired_trajectory = None
