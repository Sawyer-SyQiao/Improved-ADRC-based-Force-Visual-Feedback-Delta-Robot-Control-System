import os
import cv2
import multiprocessing
import pupil_apriltags as apriltag
import pyaudio
import audioop
import wave
import numpy as np
import time
import scipy.signal as signal
from PV_NNF import my_NNF


folder_name = "Output_MFAC"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

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


def image_processing(queue, du, y0, u1, Phi1, update_queue):
    at_detector = apriltag.Detector(families='tag36h11')
    blank_layer = np.zeros((480, 640, 3))
    path1 = np.array([[0], [0], [0], [0]])
    f = True
    n_orign = 0


    control_co_1 = [0]
    control_co_2 = [0]
    control_co_3 = [0]

    t = time.time()
    time_interval = 0
    tag_origin = np.array([0, 0, 0])

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
        time_interval = time.time() - t
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
            tag_p = tags[0].pose_t - tag_origin

            path1 = np.concatenate((path1, np.array([[tag_p[0][0]], [-tag_p[1][0]], [tag_p[2][0]], [time_interval]])),
                                   axis=1)

            n_num = n.value

            if n_num < num_cols:
                desire_pose = desired_trajectory[:, [n_num]]
                desire_pose[1][0] = -desire_pose[1][0]
            else:
                break


            controller = MFAC(du, y0, u1, Phi1)
            input = controller.control_input(desire_pose, tag_p, update_queue)

            control1 = desire_pose[0][0]*1.05 + input[0][0]
            control2 = desire_pose[1][0]*1.05 + input[1][0]
            control3 = desire_pose[2][0]*1.05 + input[2][0]

            v_error = my_NNF(np.array([[control1], [control2], [control3]]))
            control_1_row = v_error[0][0]
            control_2_row = v_error[1][0]
            control_3_row = v_error[2][0]

            control_1.value = control_1_row if abs(control_1_row) <= 1 else control_1_row / abs(control_1_row)
            control_2.value = control_2_row if abs(control_2_row) <= 1 else control_2_row / abs(control_2_row)
            control_3.value = control_3_row if abs(control_3_row) <= 1 else control_3_row / abs(control_3_row)

            cnd = blank_layer[:, :, :] > 0
            frame[cnd] = blank_layer[cnd]

        cv2.imshow('Processed Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    trigger.value = 0

    cv2.destroyAllWindows()
    file_path = os.path.join(folder_name, "detected_path_MFAC.csv")
    np.savetxt(file_path, path1, delimiter=",")


def butter_lowpass_filter(data, zi):
    global b, a
    y, zi = signal.lfilter(b, a, data, zi=zi)
    return y, zi


class MFAC:
    def __init__(self, du, y0, u1, Phi1):
        self.rho = 0.1
        self.lambda_ = 0.05
        self.m = 3
        self.u1 = u1
        self.du = du
        self.eta = 0.1
        self.mu = 1
        self.b1 = 0.001
        self.b2 = 0.5
        self.alpha = 10
        self.n = 3
        self.Phi0 = np.array([[1.1, 0, 0], [0, 0.9, 0], [0, 0, 1]])  # need to be adjusted
        self.Phi1 = Phi1
        self.y0 = y0

    def control_input(self, desire_pose, tag_p, update_queue):
        tag_p = tag_p.reshape(self.m, 1)
        desire_pose = desire_pose.reshape(self.m, 1)
        y0_np = np.frombuffer(self.y0.get_obj()).reshape((3, 3))
        Phi1_np = np.frombuffer(self.Phi1.get_obj()).reshape((3, 3))
        u1_np = np.frombuffer(self.u1.get_obj()).reshape((3, 1))
        du_np = np.frombuffer(self.du.get_obj()).reshape((3, 2))

        """
        Estimate the CFPJM matrix Phi.
        """

        Phi_update = self.eta * (((y0_np[:, 0].reshape(-1, 1) - y0_np[:, 1].reshape(-1, 1)) - Phi1_np @ du_np[:,0].reshape(-1,1)) @ du_np[:,0].reshape(1, -1)) / (self.mu + np.linalg.norm(du_np[:, 0].reshape(-1, 1), 'fro') ** 2)
        Phi = Phi1_np + Phi_update
        for i in range(self.n):
            for j in range(self.m):
                if i == j:
                    if abs(Phi[i, j]) < self.b2 or abs(Phi[i, j]) > self.alpha * self.b2 or np.sign(Phi[i, j]) != np.sign(self.Phi0[i, j]):
                        Phi[i, j] = self.Phi0[i, j]
                else:
                    if abs(Phi[i, j]) > self.b1 or np.sign(Phi[i, j]) != np.sign(self.Phi0[i, j]):
                        Phi[i, j] = self.Phi0[i, j]

        Phi1_np[:] = Phi

        """
        Compute control input u.
        """

        norm_Phi_fro = np.linalg.norm(Phi, 'fro') ** 2
        du_update = self.rho * Phi.T @ (desire_pose - y0_np[:, 0].reshape(-1, 1)) / (self.lambda_ + norm_Phi_fro)
        u = u1_np + du_update

        u1_np[:] = u

        for i in range(du_np.shape[1] - 1, 0, -1):
            du_np[:, i] = du_np[:, i - 1]
        du_np[:, 0] = du_update.flatten()

        for i in range(y0_np.shape[1] - 1, 0, -1):
            y0_np[:, i] = y0_np[:, i - 1]
        y0_np[:, 0] = tag_p.flatten()


        update_queue.put((u1_np.copy(), Phi1_np.copy(), y0_np.copy(), du_np.copy()))

        # print(f"Updated du: {du_np[:]}")
        # print(f"u1:{u1_np}")
        # print(f"Updated y: {y0_np[:]}")
        # print(f"tag_p: {tag_p}")
        # print(f"Phi1:{Phi1_np}")
        # print(f"dPhi:{Phi_update}")

        return u


def update_shared_arrays(update_queue, u1, Phi1, y0, du):
    while True:
        if not update_queue.empty():
            u1_np, Phi1_np, y0_np, du_np = update_queue.get()
            np.frombuffer(u1.get_obj())[:] = u1_np.flatten()
            np.frombuffer(Phi1.get_obj())[:] = Phi1_np.flatten()
            np.frombuffer(y0.get_obj())[:] = y0_np.flatten()
            np.frombuffer(du.get_obj())[:] = du_np.flatten()


def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    return (data, pyaudio.paContinue)


if __name__ == '__main__':
    queue = multiprocessing.Queue()
    update_queue = multiprocessing.Queue()

    control_1 = multiprocessing.Value('d', 0)
    control_2 = multiprocessing.Value('d', 0)
    control_3 = multiprocessing.Value('d', 0)
    trigger = multiprocessing.Value('i', 1)

    du = multiprocessing.Array('d', 6)
    y0 = multiprocessing.Array('d', 9)
    u1 = multiprocessing.Array('d', 3)
    Phi1 = multiprocessing.Array('d', 9)

    np.frombuffer(du.get_obj()).reshape((3, 2)).fill(0)
    np.frombuffer(y0.get_obj()).reshape((3, 3)).fill(0)
    np.frombuffer(u1.get_obj()).reshape((3, 1)).fill(0)
    np.frombuffer(Phi1.get_obj()).reshape((3, 3)).fill(0)
    # Phi1_np = np.frombuffer(Phi1.get_obj()).reshape((3, 3)).fill(0)
    # Phi1_np = np.array([[0.8, 0.0005, 0], [0, 1, 0], [0, 0, 1]])


    print("Camera Created")
    audio = pyaudio.PyAudio()
    folder = "/home/cyrus/MScProject/ProjectCode/Trajectory/Circle/"
    # folder = "/home/keke/MSc/Trajectory/circle/"
    # folder = "/home/keke/MSc/wave/ICL_v005_05/"

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

    global order, zi, b, a
    order = 2
    cutoff_freq = 10
    sampling_freq = 2387.5

    normalized_cutoff_freq = 2.0 * cutoff_freq / sampling_freq
    b, a = signal.butter(order, normalized_cutoff_freq, btype='lowpass', analog=False)

    zi1 = signal.lfiltic(b, a, 0)
    zi2 = signal.lfiltic(b, a, 0)
    zi3 = signal.lfiltic(b, a, 0)


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

    max_output = 95

    video_process = multiprocessing.Process(target=video_capture, args=(queue,))
    video_process.start()
    print("Video Started")

    image_process = multiprocessing.Process(target=image_processing, args=(queue, du, y0, u1, Phi1, update_queue))
    image_process.start()
    print("Image Processing Started")

    update_process = multiprocessing.Process(target=update_shared_arrays, args=(update_queue, u1, Phi1, y0, du))
    update_process.start()
    print("Update Process Started")

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

        data_1_co, zi1 = butter_lowpass_filter(np.array([control_1.value]), zi1)
        data_2_co, zi2 = butter_lowpass_filter(np.array([control_2.value]), zi2)
        data_3_co, zi3 = butter_lowpass_filter(np.array([control_3.value]), zi3)

        control_1.release
        control_2.release
        control_3.release

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
    stream1.close()
    stream2.close()
    wav_file1.close()
    wav_file2.close()
    wav_file3.close()
    desired_trajectory = None
