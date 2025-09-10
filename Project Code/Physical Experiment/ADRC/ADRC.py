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
from PV_NNF import my_NNF
from FV_NNF import Fz_deltaV_NNF

# ==============================
# Config
# ==============================
folder_name = "Output_ADRC"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Adaptive delay bounds (seconds) for Smith predictor
DELAY_MIN = 0.02
DELAY_MAX = 0.20
DELAY_REEVAL_STEPS = 100      # recompute delay every N control frames
DELAY_WIN_SEC = 1.8          # cross-correlation window length (seconds)
DELAY_SMOOTH_ALPHA = 0.3     # EMA smoothing on delay (0..1), higher=faster adaptation

# Nominal first-order model for Smith predictor (x_dot = -x/tau_p + b*u)
TAU_P_X = 0.06
TAU_P_Y = 0.08
TAU_P_Z = 0.06
B_P_X = 1.0
B_P_Y = 1.0
B_P_Z = 1.0
K_CORR = 0.15  # measurement correction gain for Smith predictor (0.05 ~ 0.3)

# ==============================
# Utilities
# ==============================
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

def softsat(u, umax, k=3.0):
    """Smooth saturation using tanh. k controls the 'sharpness' near limits."""
    if umax <= 0:
        return 0.0
    return float(umax * np.tanh(k * (u / umax)))

class RateLimiter:
    """Rate limiter for position-domain corrections: max |du/dt| = slew_abs_mm_per_s."""
    def __init__(self, slew_abs_mm_per_s, Ts):
        self.Ts = float(Ts)
        self._slew = float(slew_abs_mm_per_s)
        self.max_step = float(self._slew) * self.Ts
        self.y = 0.0
    def set_Ts(self, Ts):
        self.Ts = float(Ts)
        self.max_step = float(self._slew) * self.Ts
    def reset(self, y0=0.0):
        self.y = y0
    def __call__(self, x):
        dx = np.clip(x - self.y, -self.max_step, self.max_step)
        self.y = self.y + dx
        return self.y

# ==============================
# Smith Predictor (1D) with adaptive delay
# ==============================
from collections import deque

class SmithPredictor1D:
    """
    Discrete Smith predictor with online delay estimation.
    - Nominal model: x_dot = -x/tau_p + b*u (delay-free)
    - Pure delay tau_d handled by a FIFO on the 'u' path, tau_d is adapted online
    - Delay estimation uses cross-correlation between recent control proxy (r_corr)
      and measured output (y_meas), constrained in [DELAY_MIN, DELAY_MAX].
    """
    def __init__(self, Ts=1/8000, tau_p=0.06, b=1.0, k_corr=0.15,
                 delay_min=0.02, delay_max=0.25, win_sec=1.5,
                 reeval_steps=20, smooth_alpha=0.2):
        self.Ts = float(Ts)
        self.tau_p = max(1e-4, float(tau_p))
        self.b = float(b)
        self.k_corr = float(k_corr)
        self.delay_min = float(delay_min)
        self.delay_max = float(delay_max)
        self.win_sec = float(win_sec)
        self.reeval_steps = int(reeval_steps)
        self.smooth_alpha = float(smooth_alpha)

        # internal states
        self.tau_d = 0.10  # initial guess
        self._set_delay_buf()
        self.x_hat_df = 0.0
        self.x_hat_del = 0.0
        self.corr = 0.0

        # histories for delay estimation
        self.u_hist = deque(maxlen=max(10, int(self.win_sec / self.Ts)))
        self.y_hist = deque(maxlen=max(10, int(self.win_sec / self.Ts)))
        self._step = 0

    def _set_delay_buf(self):
        n_delay = max(1, int(round(self.tau_d / max(self.Ts, 1e-6))))
        self.u_buf = deque([0.0]*n_delay, maxlen=n_delay)

    def set_Ts(self, Ts):
        self.Ts = float(Ts)
        # Rebuild histories to new length
        new_len = max(10, int(self.win_sec / self.Ts))
        self.u_hist = deque(list(self.u_hist)[-new_len:], maxlen=new_len)
        self.y_hist = deque(list(self.y_hist)[-new_len:], maxlen=new_len)
        self._set_delay_buf()

    def reset(self, y0=0.0):
        self.x_hat_df = float(y0)
        self.x_hat_del = float(y0)
        self.corr = 0.0
        self._set_delay_buf()
        self.u_hist.clear()
        self.y_hist.clear()
        self._step = 0

    def _estimate_delay(self):
        """Estimate delay by maximizing normalized cross-correlation y vs u."""
        if len(self.u_hist) < 8 or len(self.y_hist) < 8:
            return None
        u = np.asarray(self.u_hist, dtype=float)
        y = np.asarray(self.y_hist, dtype=float)

        # zero-mean to avoid bias
        u = u - np.mean(u)
        y = y - np.mean(y)
        if np.linalg.norm(u) < 1e-9 or np.linalg.norm(y) < 1e-9:
            return None

        # correlation for positive lags only: y(t) vs u(t - lag)
        min_lag_samp = max(1, int(np.ceil(self.delay_min / self.Ts)))
        max_lag_samp = max(min_lag_samp, int(np.floor(self.delay_max / self.Ts)))

        # compute correlation via FFT convolution style
        # We want corr(lag) = sum_t y[t] * u[t - lag]
        corrs = []
        for lag in range(min_lag_samp, max_lag_samp+1):
            if lag >= len(u):
                corrs.append(0.0)
                continue
            u_lag = np.concatenate([np.zeros(lag), u[:-lag]])
            c = float(np.dot(y, u_lag)) / (np.linalg.norm(y) * np.linalg.norm(u_lag) + 1e-12)
            corrs.append(c)

        best_idx = int(np.argmax(corrs))
        best_lag = min_lag_samp + best_idx
        est_tau = best_lag * self.Ts
        return est_tau

    def update(self, y_meas, u_now, u_proxy_for_delay=None):
        """u_now is the controller's current 'reference correction' proxy (e.g., r_corr).
           u_proxy_for_delay, if provided, is what we log for delay estimation; default to u_now."""
        Ts = self.Ts

        # Log for delay estimation
        if u_proxy_for_delay is None:
            u_proxy_for_delay = u_now
        self.u_hist.append(float(u_proxy_for_delay))
        self.y_hist.append(float(y_meas))

        # Periodically (and cheaply) re-estimate delay
        self._step += 1
        if self._step % self.reeval_steps == 0:
            est = self._estimate_delay()
            if est is not None:
                # EMA smoothing of tau_d
                self.tau_d = (1.0 - self.smooth_alpha) * self.tau_d + self.smooth_alpha * float(est)
                # clamp within bounds
                self.tau_d = min(max(self.tau_d, self.delay_min), self.delay_max)
                # rebuild buffer if size changed
                self._set_delay_buf()

        # Delay-free update
        self.x_hat_df += Ts * (-self.x_hat_df / self.tau_p + self.b * u_now)
        # Delayed model update (use delayed control from buffer head)
        u_del = self.u_buf[0] if len(self.u_buf) > 0 else 0.0
        self.x_hat_del += Ts * (-self.x_hat_del / self.tau_p + self.b * u_del)
        # Push current control to buffer tail
        self.u_buf.append(u_now)
        # Measurement correction (align delayed-model to measured output)
        e = float(y_meas) - float(self.x_hat_del)
        self.corr += self.k_corr * e
        # Predicted current (delay-free) output
        y_pred = self.x_hat_df + self.corr
        return y_pred, self.tau_d

# ==============================
# Position-domain ADRC (TD + 3rd-order ESO + NLSEF) with SoftSat + RateLim
# ==============================
def _sign(x: float) -> float:
    return -1.0 if x < 0 else (1.0 if x > 0 else 0.0)

class ADRCRefShaper:
    """
    ADRC in position domain providing low-frequency correction 'corr' and
    output r_corr = r + corr (to feed into my_NNF). Shaping chain: SoftSat -> RateLimiter.
    """
    def __init__(self, Ts=1/8000, b0=1.0, wc=3.0, wo=12.0,
                 u_max=0.03,       # soft saturation limit (mm)
                 slew=1.0,         # max |du/dt| in mm/s
                 gamma=0.08,       # correction gain
                 soft_k=3.0):      # tanh shaping sharpness
        self.Ts = float(Ts)
        self.b0 = float(b0)
        self.wc = float(wc)
        self.wo = float(wo)
        # TD states
        self.v1 = 0.0
        self.v2 = 0.0
        # ESO states
        self.z1 = 0.0
        self.z2 = 0.0
        self.z3 = 0.0
        # Shaping
        self.u_max = float(u_max)
        self.soft_k = float(soft_k)
        self.rl = RateLimiter(slew_abs_mm_per_s=slew, Ts=Ts)
        # Bookkeeping
        self.corr_prev = 0.0
        self.last_corr = 0.0
        self.gamma = float(gamma)

    def reset(self, x0=0.0):
        self.v1 = float(x0); self.v2 = 0.0
        self.z1 = float(x0); self.z2 = 0.0; self.z3 = 0.0
        self.rl.reset(0.0)
        self.corr_prev = 0.0
        self.last_corr = 0.0

    def set_Ts(self, Ts):
        self.Ts = float(Ts)
        self.rl.set_Ts(Ts)

    # Internal TD and ESO
    def _td(self, r):
        a = self.wc
        v1_dot = self.v2
        v2_dot = -2.0*a*self.v2 - (a*a)*(self.v1 - r)
        self.v1 += self.Ts * v1_dot
        self.v2 += self.Ts * v2_dot
        return self.v1

    def _eso(self, y):
        e = self.z1 - y
        self.z1 += self.Ts * (self.z2 - 3.0*self.wo*e)
        self.z2 += self.Ts * (self.z3 + self.b0*self.corr_prev - 3.0*(self.wo**2)*e)
        self.z3 += self.Ts * (-(self.wo**3) * e)

    def update(self, r, y):
        # 1) TD
        r_td = self._td(r)
        # 2) ESO with previous corr
        self._eso(y)
        # 3) NLSEF (PD form)
        e  = r_td - self.z1
        ed = self.v2 - self.z2
        u0 = (self.wc**2) * e + 2.0*self.wc * ed
        f_hat = self.z3
        w_raw = u0 - f_hat / self.b0
        # 4) SoftSat -> RateLimiter
        w_sat = softsat(w_raw, self.u_max, self.soft_k)
        w_rl  = self.rl(w_sat)
        corr  = self.gamma * w_rl
        self.last_corr = corr
        # 5) Update corr_prev for next ESO call
        self.corr_prev = corr
        # 6) Output
        r_corr = r + corr
        return r_corr

# ==============================
# Video capture
# ==============================
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
        except:
            continue
        if not ret:
            print("Failed to read frame from camera")
            break
        queue.put(frame)
    print('Video end')
    cap.release()

# ==============================
# Image processing + control loop
# ==============================
def image_processing(queue):
    at_detector = apriltag.Detector(families='tag36h11')
    blank_layer = np.zeros((480, 640, 3))
    path1 = np.array([[0], [0], [0], [0]])
    f = True
    n_orign = 0


    # Optional mild smoothing of measured tag positions (vision-side noise only)
    alpha_x = 0.2
    alpha_y = 0.2
    alpha_z = 0.10
    lpf_x = LowPassFilter(alpha_x)
    lpf_y = LowPassFilter(alpha_y)
    lpf_z = LowPassFilter(alpha_z)

    t = time.time()
    tag_origin = np.array([0, 0, 0])

    # ===== ADRC initialisation (position domain) =====
    Ts_nom = 1/8000.0  # overwritten in loop with real period
    # Tune wc/wo and shaping per-axis
    adrc_x = ADRCRefShaper(Ts=Ts_nom, b0=2.00, wc=9.0,  wo=90.0, u_max=0.100, slew=0.30, gamma=1.0, soft_k=1.6)
    adrc_y = ADRCRefShaper(Ts=Ts_nom, b0=2.00, wc=9.0,  wo=90.0, u_max=0.100, slew=0.30, gamma=0.4, soft_k=1.6)
    adrc_z = ADRCRefShaper(Ts=Ts_nom, b0=1.40, wc=15.0,  wo=150.0, u_max=0.100, slew=0.20, gamma=0.2, soft_k=1.5)

    # ===== Smith predictors (adaptive delay) =====
    sp_x = SmithPredictor1D(Ts=Ts_nom, tau_p=TAU_P_X, b=B_P_X, k_corr=K_CORR,
                            delay_min=DELAY_MIN, delay_max=DELAY_MAX,
                            win_sec=DELAY_WIN_SEC, reeval_steps=DELAY_REEVAL_STEPS,
                            smooth_alpha=DELAY_SMOOTH_ALPHA)
    sp_y = SmithPredictor1D(Ts=Ts_nom, tau_p=TAU_P_Y, b=B_P_Y, k_corr=K_CORR,
                            delay_min=DELAY_MIN, delay_max=DELAY_MAX,
                            win_sec=DELAY_WIN_SEC, reeval_steps=DELAY_REEVAL_STEPS,
                            smooth_alpha=DELAY_SMOOTH_ALPHA)
    sp_z = SmithPredictor1D(Ts=Ts_nom, tau_p=TAU_P_Z, b=B_P_Z, k_corr=K_CORR,
                            delay_min=DELAY_MIN, delay_max=DELAY_MAX,
                            win_sec=DELAY_WIN_SEC, reeval_steps=DELAY_REEVAL_STEPS,
                            smooth_alpha=DELAY_SMOOTH_ALPHA)

    # Track last r_corr used for SP (control proxy)
    last_r_corr_x = 0.0
    last_r_corr_y = 0.0
    last_r_corr_z = 0.0

    # Timing
    t_last = time.perf_counter()

    # For logging current estimated delay
    est_delay_log = []  # [t, tau_x, tau_y, tau_z]

    while True:
        if queue.empty():
            continue
        else:
            frame = queue.get()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray_frame, estimate_tag_pose=True,
                                  camera_params=[692.299318794764, 692.371474887306, 310.832011088224,
                                                 213.144290469954],
                                  tag_size=3)

        # real sampling period for control path (~vision fps)
        t_now = time.perf_counter()
        Ts_real = t_now - t_last
        t_last = t_now
        Ts_real = max(1e-4, min(0.1, Ts_real))

        # update Ts for ADRC and SP
        adrc_x.set_Ts(Ts_real); adrc_y.set_Ts(Ts_real); adrc_z.set_Ts(Ts_real)
        sp_x.set_Ts(Ts_real); sp_y.set_Ts(Ts_real); sp_z.set_Ts(Ts_real)

        # initial origin averaging
        if f:
            if len(tags) > 0 and n_orign < 20:
                tag_origin = tags[0].pose_t
                n_orign += 1
                continue
            elif n_orign > 100:
                f = False
                continue
            elif n_orign == 20:
                print(tag_origin)
                n_orign += 1
                continue
            else:
                n_orign += 1
                continue

        if len(tags) > 0 and tags[0].tag_id == 0:
            cv2.circle(blank_layer, tuple(tags[0].center.astype(int)), 0, (0, 255, 255), 2)
            tag_p = tags[0].pose_t - tag_origin  # raw pose (meters)

            # Log path (x, -y, z, t)
            time_interval = time.time() - t
            path1 = np.concatenate((path1, np.array([[tag_p[0][0]], [-tag_p[1][0]], [tag_p[2][0]], [time_interval]])), axis=1)

            # Desired trajectory at index n (x,y,z,Fz)
            n_num = n.value
            if n_num < num_cols:
                desire_pose = desired_trajectory[0:3, [n_num]]
                desire_pose[1][0] = -desire_pose[1][0]
                desired_Fz = desired_trajectory[3, n_num]
            else:
                break

            # ====== Reset ADRC/SP when the trajectory loops back ======
            if not hasattr(image_processing, "last_n_num"):
                image_processing.last_n_num = -1
            if n_num < image_processing.last_n_num:  # looped
                adrc_x.reset(tag_p[0][0]); adrc_y.reset(tag_p[1][0]); adrc_z.reset(tag_p[2][0])
                sp_x.reset(tag_p[0][0]);   sp_y.reset(tag_p[1][0]);   sp_z.reset(tag_p[2][0])
                last_r_corr_x = tag_p[0][0]; last_r_corr_y = tag_p[1][0]; last_r_corr_z = tag_p[2][0]
                print(f"[ADRC/SP] State reset at lap, n={n_num}")
            image_processing.last_n_num = n_num

            # Optional LPF on measured pose (suppress vision noise only)
            yx = lpf_x.update(tag_p[0][0])
            yy = lpf_y.update(-tag_p[1][0])
            yz = lpf_z.update(tag_p[2][0])

            # ===== Smith predictor to estimate 'delay-free' measurements (adaptive delay) =====
            yx_sp, tau_x = sp_x.update(y_meas=yx, u_now=last_r_corr_x, u_proxy_for_delay=last_r_corr_x)
            yy_sp, tau_y = sp_y.update(y_meas=yy, u_now=last_r_corr_y, u_proxy_for_delay=last_r_corr_y)
            yz_sp, tau_z = sp_z.update(y_meas=yz, u_now=last_r_corr_z, u_proxy_for_delay=last_r_corr_z)


            est_delay_log.append([time_interval, tau_x, tau_y, tau_z])

            # ===== ADRC (position-domain) to produce corrected references =====
            r_corr_x = adrc_x.update(desire_pose[0][0], yx_sp)
            r_corr_y = adrc_y.update(desire_pose[1][0], yy_sp)
            r_corr_z = adrc_z.update(desire_pose[2][0], yz_sp)

            # Update last r_corr for SP use on next cycle
            last_r_corr_x = r_corr_x
            last_r_corr_y = r_corr_y
            last_r_corr_z = r_corr_z

            # ===== Feed corrected reference to my_NNF (trajectory -> normalized voltages) =====
            nnf_input = np.array([[r_corr_x], [r_corr_y], [r_corr_z]], dtype=float)

            # Force compensation (if desired_Fz != 0)
            if abs(desired_Fz) < 1e-6:
                v_error = my_NNF(nnf_input)
            else:
                input_with_fz = np.vstack([nnf_input, [[desired_Fz]]])
                v_error = my_NNF(nnf_input) + Fz_deltaV_NNF(input_with_fz)

            # Write shared controls (clip to [-1, 1])
            control_1_row = float(np.clip(v_error[0][0], -1.0, 1.0))
            control_2_row = float(np.clip(v_error[1][0], -1.0, 1.0))
            control_3_row = float(np.clip(v_error[2][0], -1.0, 1.0))
            control_1.value = control_1_row
            control_2.value = control_2_row
            control_3.value = control_3_row

            # Overlay
            cnd = blank_layer[:, :, :] > 0
            frame[cnd] = blank_layer[cnd]

        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    trigger.value = 0
    cv2.destroyAllWindows()
    file_path = os.path.join(folder_name, "detected_path_ADRC.csv")
    np.savetxt(file_path, path1, delimiter=",")
    # save delay estimation log
    file_path = os.path.join(folder_name, "adaptive_delay_log.csv")
    np.savetxt(file_path, np.asarray(est_delay_log), delimiter=",")

# ==============================
# Audio path + main
# ==============================
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
    folder = "/home/cyrus/MScProject/ProjectCode/ADRC/z_direction/"
    wav_file1_name = folder + "control01.wav"
    wav_file1 = wave.open(wav_file1_name, 'rb')
    wav_file2_name = folder + "control01.wav"
    wav_file2 = wave.open(wav_file2_name, 'rb')
    wav_file3_name = folder + "control01.wav"
    wav_file3 = wave.open(wav_file3_name, 'rb')

    print("Voltage Data Imported")

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

    global order, zi, b, a, b2, a2, duration, num_channels
    order = 2
    cutoff_freq = 2
    sampling_freq = 1000
    normalized_cutoff_freq = 2.0 * cutoff_freq / sampling_freq
    b, a = signal.butter(order, normalized_cutoff_freq, btype='lowpass', analog=False)

    zi1 = signal.lfiltic(b, a, 0)
    zi2 = signal.lfiltic(b, a, 0)
    zi3 = signal.lfiltic(b, a, 0)

    # Audio-side gentle smoothing of voltages (unchanged)
    alpha = 0.2
    low_pass_filter1 = LowPassFilter(alpha)
    low_pass_filter2 = LowPassFilter(alpha)
    low_pass_filter3 = LowPassFilter(alpha)

    print("Filter Created")

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

    FACTOR1 = 1
    FACTOR2 = 1
    FACTOR3 = 1

    video_process = multiprocessing.Process(target=video_capture, args=(queue,))
    video_process.start()
    print("Video Started")

    image_process = multiprocessing.Process(target=image_processing, args=(queue,))
    image_process.start()
    print("Image Processing Started")

    time.sleep(10)
    n_x = 0

    data_1_co = [0]
    data_2_co = [0]
    data_3_co = [0]
    data_1_co_former = [0]
    data_2_co_former = [0]
    data_3_co_former = [0]
    control_co_1 = [0]
    control_co_2 = [0]
    control_co_3 = [0]

    # start with small non-zero to satisfy PyAudio write
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
    t = time.time()
    while n.value < num_cols:
        data_1_co_former = data_1_co
        data_2_co_former = data_2_co
        data_3_co_former = data_3_co

        # read shared controls
        with control_1.get_lock():
            v1_val = control_1.value
        with control_2.get_lock():
            v2_val = control_2.value
        with control_3.get_lock():
            v3_val = control_3.value

        # audio-side low-pass
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

    print(f"Playing stops")
    file_path = os.path.join(folder_name, "control_co_1.csv")
    np.savetxt(file_path, control_co_1, delimiter=",")
    file_path = os.path.join(folder_name, "control_co_2.csv")
    np.savetxt(file_path, control_co_2, delimiter=",")
    file_path = os.path.join(folder_name, "control_co_3.csv")
    np.savetxt(file_path, control_co_3, delimiter=",")
    # 保存参考轨迹

    stream1.close()
    stream2.close()
    wav_file1.close()
    wav_file2.close()
    wav_file3.close()
    desired_trajectory = None




