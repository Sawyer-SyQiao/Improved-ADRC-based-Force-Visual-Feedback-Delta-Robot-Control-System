import cv2
import multiprocessing
import pupil_apriltags as apriltag
import pyaudio
import audioop
import wave
import numpy as np
import time
import scipy.signal as signal
from Siyuan_NNF_0703 import my_NNF


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


def video_capture(queue):
    # Create a VideoCapture object to capture video from the camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index if multiple cameras are connected
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # Check if the camera was opened successfully
    if not cap.isOpened():
        print("Failed to open camera")
        return


    while trigger.value:
        # Read a frame from the camera
        try:
            ret, frame = cap.read()
        except:
            continue

        # Check if the frame was successfully read
        if not ret:
            print("Failed to read frame from camera")
            break

        # Put the frame into the queue for image processing
        queue.put(frame)

    # Release the VideoCapture object
    print('Video end')
    cap.release()

def image_processing(queue):
    ##在这里
    import datetime
    start_time_real = None
    end_time_real = None
    delta_t = 1 / sampling_freq
    counter = 0

    at_detector = apriltag.Detector(families='tag36h11')
    blank_layer = np.zeros((480, 640, 3))
    path1 = np.array([[0],[0],[0],[0]])
    f = True
    n_orign = 0

    alpha = 0.2
    low_pass_filter4 = LowPassFilter(0.2)
    low_pass_filter5 = LowPassFilter(0.2)
    low_pass_filter6 = LowPassFilter(0.2)

    
    control_co_1 = [0]
    control_co_2 = [0]
    control_co_3 = [0]

    control_1_row = 0
    control_2_row = 0
    control_3_row = 0

    former_pid_1 = 0
    former_pid_2 = 0
    former_pid_3 = 0
    control_min = 0.01

    t = time.time()
    time_interval = 0;
    tag_origin = np.array([[0],[0],[0]])
    force_range = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    voltage_diff = np.array([[0],[0],[0]])

    while True:
        # Get a frame from the queue for processing
        if queue.empty():
            continue
        else:
            frame = queue.get()

        # Perform image processing on the frame (example: convert to grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = at_detector.detect(gray_frame,estimate_tag_pose = True,
                            camera_params = [692.299318794764,692.371474887306,310.832011088224,213.144290469954],
                            tag_size = 3)
        time_interval = time.time()-t
        if f:
            if len(tags)>0 and n_orign < 100 :
                tag_origin = np.add(tag_origin,tags[0].pose_t)
                # Shot down the judgement after the original position is obtained
                n_orign = n_orign + 1
                continue
            elif n_orign>100:
                f = False
                continue
            elif n_orign == 100:
                tag_origin = np.divide(tag_origin,n_orign)
                print(tag_origin)
                n_orign = n_orign + 1
                continue
            else:
                # If not tags detected, original position is [0,0,0]
                n_orign = n_orign + 1
                continue
        
        if len(tags)>0 and tags[0].tag_id==0:

##        for tag in tags:
##            if tag.tag_id == 0:
##                
##                cv2.circle(blank_layer, tuple(tag.center.astype(int)), 0, (0, 255, 255), 2)
##                tag_p = tag.pose_t
##                #print(tag.center.astype(int))
            cv2.circle(blank_layer, tuple(tags[0].center.astype(int)), 0, (0, 255, 255), 2)
            tag_p = tags[0].pose_t-tag_origin
##            
            path1 = np.concatenate((path1,np.array([[tag_p[0][0]],[-tag_p[1][0]],[tag_p[2][0]],[time_interval]])), axis = 1)

##            voltage_diff = np.array([[control_1_row*max_output],[control_2_row*max_output],[control_3_row*max_output]]) - my_NNF(np.array([[tag_p[0][0]],[-tag_p[1][0]],[tag_p[2][0]]]))*max_output

            
            voltage_diff = np.array([[control_force_1.value*max_output],[control_force_2.value*max_output],[control_force_3.value*max_output]]) - my_NNF(np.array([[tag_p[0][0]],[tag_p[1][0]],[tag_p[2][0]]]))*max_output
            force_value = np.matmul(tm_force, voltage_diff)
##            voltage_diff = my_NNF(np.array([[tag_p[0][0]],[tag_p[1][0]],[tag_p[2][0]]]))*max_output
##            force_value = np.array([[control_force_1.value*max_output],[control_force_2.value*max_output],[control_force_3.value*max_output],[voltage_diff[0][0]],[voltage_diff[1][0]],[voltage_diff[2][0]]])

            force_range = np.concatenate((force_range,np.array([[time_interval],[force_value[0][0]],[force_value[1][0]],[force_value[2][0]],[control_force_1.value*max_output],[control_force_2.value*max_output],[control_force_3.value*max_output],[voltage_diff[0][0]],[voltage_diff[1][0]],[voltage_diff[2][0]]])), axis = 1)
            
            ##在这里
            current_time = counter * delta_t
            voltage_diff_list.append([current_time, voltage_diff[0][0], voltage_diff[1][0], voltage_diff[2][0]])
            counter += 1
            if start_time_real is None:
                start_time_real = datetime.datetime.now()
            end_time_real = datetime.datetime.now()

            
            n_num  = n.value
            
            if n_num < num_cols:
##                desire_pose = desired_trajectory[:,[n_num]]*0.85/5*5*1.3
                desire_pose = desired_trajectory[:,[n_num]]
                
##                desire_pose = np.array([[0.3],[0],[0]])
##                a_inter = desire_pose[0][0]
##                desire_pose[0][0] = desire_pose[2][0]
##                desire_pose[2][0] = a_inter 
                
                desire_pose[1][0] = -desire_pose[1][0]
##                desire_pose[0][0] =  2*desire_pose[0][0]
                

            else:
                break
##            if tag_p[0][0]>=0:
##                tag_p[0][0] = low_pass_filter4.update(tag_p[0][0]*1.3)
##            else:
##                tag_p[0][0] = low_pass_filter4.update(tag_p[0][0]*1.5)
##
##            if tag_p[1][0]<0:
##                tag_p[1][0] = low_pass_filter4.update(tag_p[1][0])
##            else:
##                tag_p[1][0] = low_pass_filter4.update(tag_p[1][0]*1.3)
                
 
            tag_p[0][0] = low_pass_filter4.update(tag_p[0][0])
            tag_p[1][0] = low_pass_filter5.update(tag_p[1][0])
            tag_p[2][0] = low_pass_filter6.update(tag_p[2][0])            
            error_pose = desire_pose - tag_p
            
##            if abs(error_pose[0] - former_pid_1)<control_min and error_pose[0] - former_pid_1 != 0:
##                error_pose[0] = former_pid_1 + (error_pose[0]-former_pid_1)/abs(error_pose[0]-former_pid_1)*control_min
##            former_pid_1 = error_pose[0] 
##
##            if abs(error_pose[1] -former_pid_2)<control_min and error_pose[1] - former_pid_2 != 0:
##                error_pose[1] = former_pid_2 + (error_pose[1]-former_pid_2)/abs(error_pose[1]-former_pid_2)*control_min
##            former_pid_2 = error_pose[1] 
##
##            if abs(error_pose[2]-former_pid_3)<control_min and error_pose[2] - former_pid_3 != 0:
##                error_pose[2] = former_pid_3 + (error_pose[2] - former_pid_3)/abs(error_pose[2]-former_pid_3)*control_min
##            former_pid_3 = error_pose[2]



            if pidtrigger.value:

                control1 = 0 # X directionPID control
                control2 = 0  # Y directionPID control
                control3 = 0 # Z directionPID control
            else:
                control1 = desire_pose[0][0]*0.9 + piezo_pid1.update(-error_pose[0][0])# X directionPID control
                control2 = desire_pose[1][0]*0.9 + piezo_pid2.update(-error_pose[1][0]) # Y directionPID control
                control3 = desire_pose[2][0]*0.9 + piezo_pid3.update(-error_pose[2][0]) # Z directionPID control



##            0.8

##            v_error = np.matmul(tm_inv,np.array([[control1],[control2],[control3]]))*2
            v_error = my_NNF(np.array([[control1],[control2],[control3]]))
            control_1_row = v_error[0][0]
            control_2_row = v_error[1][0]
            control_3_row = v_error[2][0]



            if abs(control_1_row) > 1:
                control_1.value = control_1_row/abs(control_1_row)
            else:
                control_1.value = control_1_row
            if abs(control_2_row) > 1:
                control_2.value = control_2_row/abs(control_2_row)
            else:
                control_2.value = control_2_row
            if abs(control_3_row) > 1:
                control_3.value = control_3_row/abs(control_3_row)
            else:
                control_3.value = control_3_row

            
            
            
##                
##            control_1.value = 0.9*np.sin(2*3.1415*(time.time()-t))
##            control_2.value = 1*np.sin(2*3.1415*(time.time()-t))
##            control_3.value = 0.9*np.sin(2*3.1415*(time.time()-t))
####            control_2.value = 0.9

##

##            control_co_1.append(v_error[0][0]) 
##            control_co_2.append(v_error[1][0])
##            control_co_3.append(v_error[2][0])


            cnd = blank_layer[:, :, :] > 0
            frame[cnd] = blank_layer[cnd]
            # Display the processed frame

        cv2.imshow('Processed Video', frame)

        # Wait for the 'q' key to be pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the window
    trigger.value = 0
    
    cv2.destroyAllWindows()
    np.savetxt("detected_path.csv", path1, delimiter=",")
    np.savetxt("detected_force.csv", force_range, delimiter=",")
##    np.savetxt("control_co_1.csv", control_co_1, delimiter=",")
##    np.savetxt("control_co_2.csv", control_co_2, delimiter=",")
##    np.savetxt("control_co_3.csv", control_co_3, delimiter=",")

    ##在这里
    voltage_diff_array = np.array(voltage_diff_list).T
    np.savetxt("voltage_diff.csv", voltage_diff_array, delimiter=",")
    with open("voltage_diff_time_range.txt", "w") as f:
        f.write(f"StartTime: {start_time_real.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}\n")
        f.write(f"EndTime:   {end_time_real.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}\n")


##    
def butter_lowpass_filter(data, zi):
    # Apply the filter to the data
    global b, a
    y, zi = signal.lfilter(b, a, data, zi=zi)

    # Return the filtered data and the updated filter state
    return y, zi


class PID:
    def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10):
        self.Kp=P
        self.Ki=I
        self.Kd=D
        self.Derivator=Derivator
        self.Integrator=Integrator
        self.Integrator_max=Integrator_max
        self.Integrator_min=Integrator_min

        self.set_point=0.0
        self.error=0.0

    def update(self,current_value):
        """
        Calculate PID output value for given reference input and feedback
        """

        self.error = self.set_point - current_value


##        self.Kp += 0.001 * self.error
##        self.Ki += 0.0001 * self.Integrator
##        self.Kd += 0.001 * (self.error - self.Derivator)
        


        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * ( self.error - self.Derivator)
        self.Derivator = self.error

        if abs(self.error)<=0.1:
            self.Integrator = self.Integrator + self.error
        


        if self.Integrator > self.Integrator_max:
                self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
                self.Integrator = self.Integrator_min

        self.I_value = self.Integrator * self.Ki

        PID = self.P_value + self.I_value + self.D_value

        return PID

    def setPoint(self,set_point):
        """
        Initilize the setpoint of PID
        """
        self.set_point = set_point
        self.Integrator=0
        self.Derivator=0

    def setIntegrator(self, Integrator):
        self.Integrator = Integrator

    def setDerivator(self, Derivator):
        self.Derivator = Derivator

    def setKp(self,P):
        self.Kp=P

    def setKi(self,I):
        self.Ki=I

    def setKd(self,D):
        self.Kd=D

    def getPoint(self):
        return self.set_point

    def getError(self):
        return self.error

    def getIntegrator(self):
        return self.Integrator

    def getDerivator(self):
        return self.Derivator

def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        # If len(data) is less than requested frame_count, PyAudio automatically
        # assumes the stream is finished, and the stream stops.
        return (data, pyaudio.paContinue)


if __name__ == '__main__':
    # Create a queue for communication between the processes
    queue = multiprocessing.Queue()
    

    control_1 = multiprocessing.Value('d',0)
    control_2 = multiprocessing.Value('d',0)
    control_3 = multiprocessing.Value('d',0)

    control_force_1 = multiprocessing.Value('d',0)
    control_force_2 = multiprocessing.Value('d',0)
    control_force_3 = multiprocessing.Value('d',0)
    trigger = multiprocessing.Value('i',1);
    pidtrigger = multiprocessing.Value('i',1);
    

##    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index if multiple cameras are connected
##    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # Transfer Matrix
    
    tm =np.array( [[-5.72078325980938e-05, -0.00410628775442385, 0.00293996393087933], 
                   [0.00307682724989050, -0.00185656986031614, -0.00196658738779843],
                   [0.00222411684809831, 0.00334654417309416, 0.00321324400929087]]) #line

##    tm_force =np.array([[-3.71, 14.76, -18.02], [-11.79, 4.59, 4.93],[-11.26, 13.90, 14.49]])/100 #linep
    tm_force =np.array([[0.19, 9.23, -13.08], [-9.21,5.86,5.17],[7.58,8.31,7.34]])/100 #linep
    
    tm_inv = np.linalg.inv(tm)

    # PID controller

        # Create a pid with Kp, Ki and Kd entered
##    Kp = 0.8
##    Ki = 0
##    Kd = 0.1
##  500mg
##    Kp = 0.8
##    Ki = 0.005
##    Kd = 0.3

    ##  0-200mg
##    Kp = 0.8
##    Ki = 0.05
##    Kd = 0.02
    Kp = 0.95
    Ki = 0.0025
    Kd = 0.01
##
##    Kp = 0.0
##    Ki = 0.0
##    Kd = 0.0    

    
    piezo_pid1 = PID()
    piezo_pid2 = PID()
    piezo_pid3 = PID()
    
    piezo_pid1.setKp(Kp)
    piezo_pid2.setKp(Kp)
    piezo_pid3.setKp(Kp)
    
    piezo_pid1.setKi(Ki)
    piezo_pid2.setKi(Ki)
    piezo_pid3.setKi(Ki)
    
    piezo_pid1.setKd(Kd)
    piezo_pid2.setKd(Kd)
    piezo_pid3.setKd(Kd)
    
    piezo_pid1.setPoint(0)
    piezo_pid2.setPoint(0)
    piezo_pid3.setPoint(0)


    
    print("Camera Created")
    audio = pyaudio.PyAudio()
##    folder = "/home/md2023/wave_13/real_control/Real_circle_v05_04/"
##    folder = "/home/md2023/wave_13/2023_03_20_spiral/"
##    folder = "/home/md2023/wave_13/Cir_v01_05_1221/
##    folder = "/home/md2023/wave_13/lift_v05_1_0626/"
##    folder = "/home/md2023/wave_13/circle_v2_05_0626/"
##    folder = "/home/md2023/wave_13/new_sq_v02_04/"
##    folder = "/home/md2023/wave_13/sq_res/Res_v01_resq/"
##    folder = "/home/md2023/wave_13/ICL_v005_05/"
##    folder = "/home/md2023/wave_13/resolution/Res_v01_sqy03/"
##    folder = "/home/md2023/wave_13/3D_res/Res_v01_3Dsq_030/"
##    folder = "/home/md2023/wave_13/Heart_v01/"
##    folder = "/home/md2023/wave_13/Pre&Acc/Pre5_v01_01/"
##    folder = "/home/md2023/wave_13/lift_v05_05_0224/"
##    folder = "/home/md2023/wave_13/lift_v05_05_250521/"
    folder = "/home/cyrus/MSc Project/ProjectCode/Trajectory/z_direction/"

    

    wav_file1_name = folder + "control01.wav"
    wav_file1 = wave.open(wav_file1_name, 'rb')
    wav_file2_name = folder + "control01.wav"
    wav_file2 = wave.open(wav_file2_name, 'rb')
    wav_file3_name = folder + "control01.wav"
    wav_file3 = wave.open(wav_file3_name, 'rb')

    print("Voltage Data Imported")

    device_index1 = 3 # Change this to the desired No.1 device index 
    device_info1 = audio.get_device_info_by_index(device_index1)

    device_index2 = 2 # Change this to the desired No.2 device index
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

    # Set the filter parameters
    global order, zi, b, a, b2, a2, duration, num_channels
    order = 2 # Filter order
    cutoff_freq = 10 # Cutoff frequency (Hz)
    sampling_freq = 2387.5 # Sampling frequency (Hz)
##    sampling_freq = 8000 # Sampling frequency (Hz)

    ##在这里
    voltage_diff_list = [] 
    counter = 0
    delta_t = 1 / sampling_freq

    

    # Calculate the normalized cutoff frequency
    normalized_cutoff_freq = 2.0 * cutoff_freq / sampling_freq
  
    # Create the filter coeffqicients using Butterworth filter design
    b, a = signal.butter(order, normalized_cutoff_freq, btype='lowpass',analog = False)
    

    zi1 = signal.lfiltic(b, a, 0)
    zi2 = signal.lfiltic(b, a, 0)
    zi3 = signal.lfiltic(b, a, 0)

    alpha = 0.2
    low_pass_filter1 = LowPassFilter(alpha)
    low_pass_filter2 = LowPassFilter(alpha)
    low_pass_filter3 = LowPassFilter(alpha)


    print("Filter Created")

    # Import desired tarjectory points
    
    trajectory_file_name = folder + "desired_path.csv"
    desired_trajectory = np.genfromtxt(trajectory_file_name, delimiter=',')
    stop_file_name = folder + "stop.csv"
    stop_trajectory = np.genfromtxt(stop_file_name, delimiter=',')
    

    num_rows, num_cols = desired_trajectory.shape


    
    n = multiprocessing.Value('i',0);
    
    

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

    # Create and start the video capture process
    video_process = multiprocessing.Process(target=video_capture, args=(queue,))
    video_process.start()
    print("Video Started")

    # Create and start the image processing process
    image_process = multiprocessing.Process(target=image_processing, args=(queue,))
    image_process.start()
    print("Image Processing Started")

    # Wait for the processes to complete
##    video_process.join()
##    image_process.join()

    
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

    voltage_diff_1 = [0]
    voltage_diff_2 = [0]
    voltage_diff_3 = [0]

    data1 = b'\xff\x7f'*samples
    data2 = b'\xff\x7f'*samples
    data3 = b'\xff\x7f'*samples

    data_row = b'\xff\x7f'

    
    
    data1 = audioop.mul(data1, WIDTH1, data_1_co[0])
    data2 = audioop.mul(data2, WIDTH2, data_2_co[0])
    data3 = audioop.mul(data3, WIDTH3, data_3_co[0])

    data_2 = np.frombuffer(data2, dtype=np.int16).reshape(-1,1)
    data_3 = np.frombuffer(data3, dtype=np.int16).reshape(-1,1)
    data_4 = np.concatenate((data_2,data_3), axis = 1)

    data2_2 = data_4.tobytes()
    data1_2 = data1
    pidtrigger.value = 0
    print("Voltage Supply Started")
    t= time.time()
    while n.value < num_cols:
        
        data_1_co_former = data_1_co

        data_2_co_former = data_2_co

        data_3_co_former = data_3_co
        

        control_1.get_lock()
        control_2.get_lock()
        control_3.get_lock()

        data_1_co, zi1 = butter_lowpass_filter(np.array([control_1.value]), zi1)
        data_2_co, zi2 = butter_lowpass_filter(np.array([control_2.value]), zi2)
        data_3_co, zi3 = butter_lowpass_filter(np.array([control_3.value]), zi3)
        
        control_1.release
        control_2.release
        control_3.release

        data_1_co[0] = low_pass_filter1.update(data_1_co[0])
        data_2_co[0] = low_pass_filter2.update(data_2_co[0])
        data_3_co[0] = low_pass_filter3.update(data_3_co[0])

        control_force_1.get_lock()
        control_force_2.get_lock()
        control_force_3.get_lock()

        control_force_1.release
        control_force_2.release
        control_force_3.release
        
##        filtered_value1 = low_pass_filter1.update(control_1.value)
##        filtered_value2 = low_pass_filter2.update(control_2.value)
##        filtered_value3 = low_pass_filter3.update(control_3.value)
##

        

####
##        data_1_co, zi4 = butter_lowpass_filter2(data_1_co, zi4)
##        data_2_co, zi5 = butter_lowpass_filter2(data_2_co, zi5)
##        data_3_co, zi6 = butter_lowpass_filter2(data_3_co, zi6)

##        control_co_1.append(control_1.value)
##        control_co_2.append(control_2.value)
##        control_co_3.append(control_3.value)
##    

        
##        data1 = audioop.mul(data1, WIDTH1, FACTOR1)
##        data2 = audioop.mul(data2, WIDTH2, FACTOR2)
##        data3 = audioop.mul(data3, WIDTH3, FACTOR3)

##        data1 = audioop.mul(data1, WIDTH1, data_1_co[0])
##        data2 = audioop.mul(data2, WIDTH2, data_2_co[0])
##        data3 = audioop.mul(data3, WIDTH3, data_3_co[0])


                    



        stream1.write(data1_2,  samples, exception_on_underflow=True)
        stream2.write(data2_2,  samples, exception_on_underflow=True)
##
##        stream1.write(data1_2,  samples, exception_on_underflow=True)
##        stream2.write(data2_2,  samples, exception_on_underflow=True)
        



        



        



##        print(t)



        
##
##        n.value += samples
##        data1 = wav_file1.readframes(samples)
##        data2 = wav_file2.readframes(samples)
##        data3 = wav_file3.readframes(samples)
        n_x += 1#
        #if n_x >=4:
        if n_x >=2:

            # Move the iterator forward


            data1 = audioop.mul(data_row, WIDTH1, data_1_co_former[0]) + audioop.mul(data_row, WIDTH1, data_1_co[0]) 

            data2 = audioop.mul(data_row, WIDTH2, data_2_co_former[0]) + audioop.mul(data_row, WIDTH2, data_2_co[0])

            data3 = audioop.mul(data_row, WIDTH3, data_3_co_former[0]) + audioop.mul(data_row, WIDTH3, data_3_co[0])

            
            


            
##            data1 = audioop.mul(data1, WIDTH, data_1_co[0])
##            data2 = audioop.mul(data2, WIDTH, data_2_co[0])
##            data3 = audioop.mul(data3, WIDTH, data_3_co[0])


##            data1 = audioop.mul(data1, WIDTH1, filtered_value1)
##            data2 = audioop.mul(data2, WIDTH2, filtered_value2)
##            data3 = audioop.mul(data3, WIDTH3, filtered_value3)
##
##
##            control_co_1.append(filtered_value1)
##            control_co_2.append(filtered_value2)
##            control_co_3.append(filtered_value3)  

            control_co_1.append(data_1_co[0])
            control_co_2.append(data_2_co[0])
            control_co_3.append(data_3_co[0])

            
##            data1 = audioop.mul(data1, WIDTH1, control_1.value)
##            data2 = audioop.mul(data2, WIDTH2, control_2.value)
##            data3 = audioop.mul(data3, WIDTH3, control_3.value)

            data_2 = np.frombuffer(data2, dtype=np.int16).reshape(-1,1)
            data_3 = np.frombuffer(data3, dtype=np.int16).reshape(-1,1)
            data_4 = np.concatenate((data_2,data_3), axis = 1)

            data2_2 = data_4.tobytes()
            data1_2 = data1
                    


            n.value += 1
            #n.value += 2

##            if n.value in stop_trajectory:
##                time.sleep(1)
##            n_x  = 0


        


    print(f"Playing stops")
    np.savetxt("control_co_1.csv", control_co_1, delimiter=",")
    np.savetxt("control_co_2.csv", control_co_2, delimiter=",")
    np.savetxt("control_co_3.csv", control_co_3, delimiter=",")
    stream2.close()
    wav_file1.close()
    wav_file2.close()
    wav_file3.close()
    desired_trajectory = None


