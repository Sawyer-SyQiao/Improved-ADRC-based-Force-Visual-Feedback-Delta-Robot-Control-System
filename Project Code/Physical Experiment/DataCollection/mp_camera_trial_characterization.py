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
import sounddevice as sd
from pyDOE import lhs

def butter_lowpass_filter(data, zi):
    # Apply the filter to the data
    global b, a
    y, zi = signal.lfilter(b, a, data, zi=zi)

    # Return the filtered data and the updated filter state
    return y, zi

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
    else:
        print("Camera Opened")

    while trigger.value:
        # Read a frame from the camera
        ret, frame = cap.read()


        # Check if the frame was successfully read
        if not ret:
            print("Failed to read frame from camera")
            break

        # Put the frame into the queue for image processing
        if queue.empty():
            if queue.put(frame):
                pass
            else:
                continue
        else:
            continue
  

    # Release the VideoCapture object
    print('Video end')
    cap.release()

def image_processing(queue,queue_tag_p):

    at_detector = apriltag.Detector(families='tag36h11',nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)
    blank_layer = np.zeros((480, 640, 3))
    print("Processing Start")
##    path1 = np.array([[0],[0],[0],[0],[0],[0]])
    f = True
    f2 = True
    n_orign = 0

    order1 = 2  # Filter order
    cutoff_freq1 = 10 # Cutoff frequency (Hz)
    sampling_freq1 = 30  # Sampling frequency (Hz)

    # Calculate the normalized cutoff frequency
    normalized_cutoff_freq1 = 2.0 * cutoff_freq1 / sampling_freq1

    # Create the filter coefficients using Butterworth filter design
    b1, a1 = signal.butter(order1, normalized_cutoff_freq1, btype='lowpass')
    zi4 = signal.lfiltic(b1, a1, 0)
    trigger.value = True
    
    while trigger.value:
        # Get a frame from the queue for processing
        if queue.empty():
            continue
        else:
            try:
                frame = queue.get()
            except queue.Empty:
                continue
        # Perform image processing on the frame (example: convert to grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

##        tags = at_detector.detect(gray_frame,estimate_tag_pose = True,
##                            camera_params = [660.7810,680.0870,308.6991,281.2018],
##                            tag_size = 3)
        tags = at_detector.detect(gray_frame,estimate_tag_pose = True,
                    camera_params = [692.2993,692.3714,310.8320,213.1442],
                    tag_size = 3)
        if f:
            if len(tags)>0 and n_orign> 5 :
                tag_origin = tags[0].pose_t
                f = False                  # Shot down the judgement after the original position is obtained
                print(tag_origin)
            else:
                tag_origin = np.array([0,0,0]) # If not tags detected, original position is [0,0,0]
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

            n_num  = n.value
            if n_num < num_cols:
                desire_pose = desired_trajectory[:,[n_num]]
            else:
                break
##            data_4_co, zi4 = butter_lowpass_filter(np.array([tag_p[2][0]]), zi4)
##
##            tag_p[2][0] = data_4_co[0]
            
            error_pose = 2*desire_pose-tag_p

##            error_pose[2][0] = error_pose[0][0]
            error_pose[0][0] = -error_pose[0][0]
##            error_pose[1][0] = -error_pose[1][0]

            
##            error_pose[2][0] = 0

            

            control1 = piezo_pid1.update(error_pose[0][0])# X directionPID control
            control2 = piezo_pid2.update(error_pose[1][0]) # Y directionPID control
            control3 = piezo_pid3.update(error_pose[2][0]) # Z directionPID control

           

            # v_error = np.matmul(tm_inv,np.array([[control1],[control2],[control3]]))

            v_error = my_NNF(np.array([[control1],[control2],[control3]]))
            control_1_row = v_error[0][0]/max_output
            control_2_row = v_error[1][0]/max_output
            control_3_row = v_error[2][0]/max_output

            
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

##            path1 = np.concatenate((path1,tag_p, axis = 1)
            
            cnd = blank_layer[:, :, :] > 0
            frame[cnd] = blank_layer[cnd]

            if f2 and n_num >= 32000:
                if queue_tag_p.empty():
                    queue_tag_p.put(tag_p)
                    f2 = False
                else:
                    continue

        else:
            continue
            
            # Display the processed frame

        cv2.imshow('Processed Video', frame)

        # Wait for the 'q' key to be pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the window
    trigger.value = 0
    

##    np.savetxt("detected_path.csv", path1, delimiter=",")



class PID:
    def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0, Integrator=0, Integrator_max=500, Integrator_min=-500):
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

        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * ( self.error - self.Derivator)
        self.Derivator = self.error

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


if __name__ == '__main__':
    # Create a queue for communication between the processes
    


    control_1 = multiprocessing.Value('d',0);
    control_2 = multiprocessing.Value('d',0);
    control_3 = multiprocessing.Value('d',0);
    


##    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index if multiple cameras are connected
##    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # Transfer Matrix
    
    tm =np.array( [[-5.72078325980938e-05, -0.00410628775442385, 0.00293996393087933], 
                   [0.00307682724989050, -0.00185656986031614, -0.00196658738779843],
                   [0.00222411684809831, 0.00334654417309416, 0.00321324400929087]]) #line
    
    tm_inv = np.linalg.inv(tm)

    # PID controller

        # Create a pid with Kp, Ki and Kd entered
    Kp = 0.6
    Ki = 0
    Kd = 0.05

    

    
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


    max_output = 9

    max_output = 95
    num_samples = 100  # 只需设置总采样点数

    lhs_points = lhs(3, samples=num_samples, criterion='maximin')  # 3维
    Factors = lhs_points * 2 - 1    # 映射到[-1,1]区间
    Factors_volt = Factors * max_output  # 映射到[-95,95]

    # Factors.shape = (num_samples, 3)，如需和原代码兼容可转置
    Factors = Factors.T  # 变为(3, num_samples)
    path = np.array([[0],[0],[0],[0],[0],[0]])
    factor_counter = 0


    # Create a factor array
    # step = 38
    # step = 76
#     num_points = 5
    
    
#     # Factor1_array = np.array(range(-max_output, max_output+step, step))/max_output
#     Factor1_array = np.linspace(-max_output, max_output, num_points) / max_output
#     # Factor2_array = np.array(range(-max_output, max_output+step, step))/max_output
#     Factor2_array = np.linspace(-max_output, max_output, num_points) / max_output
# ##    Factor2_array = np.array([-1,1])
    
#     # Factor3_array = np.array(range(-max_output, max_output+step, step))/max_output
#     Factor3_array = np.linspace(-max_output, max_output, num_points) / max_output
    
#     Factors  = np.array([[0],[0],[0]])

#     for f1 in Factor1_array:
#         for f2 in Factor2_array:
#             for f3 in Factor3_array:
#                 Factors = np.concatenate((Factors,np.array([[f1],[f2],[f3]])), axis = 1)

    Facter_rows, Factor_cols = Factors.shape

    for Factor_index in range(Factor_cols):
        queue_tag_p = multiprocessing.Queue()
        queue = multiprocessing.Queue()
        trigger = multiprocessing.Value('i',1);

        FACTOR1,  FACTOR2,  FACTOR3 = Factors[:, Factor_index]

        trigger.Value = 1
    
        folder = "/home/cyrus/MScProject/ProjectCode/char/"
        wav_file1_name = folder + "control_char.wav"
        wav_file1 = wave.open(wav_file1_name, 'rb')
        wav_file2_name = folder + "control_char.wav"
        wav_file2 = wave.open(wav_file2_name, 'rb')
        wav_file3_name = folder + "control_char.wav"
        wav_file3 = wave.open(wav_file3_name, 'rb')

        print("Voltage Data Imported")

##        Device_Dic = sd.query_devices()
##
##        for Device_index in range(len(Device_Dic)):
##            if Device_Dic[Device_index]['name'] == 'Boreas DevKit (BOS1901): USB Audio (hw:1,0)':
##                device_index1 = Device_index
##
##            if Device_Dic[Device_index]['name'] == 'Boreas DevKit (BOS1901): USB Audio (hw:2,0)':
##                device_index2 = Device_index

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

        # Set the filter parameters
       
        order = 2  # Filter order
        cutoff_freq = 10 # Cutoff frequency (Hz)
        sampling_freq = wav_file1.getframerate()  # Sampling frequency (Hz)

        # Calculate the normalized cutoff frequency
        normalized_cutoff_freq = 2.0 * cutoff_freq / sampling_freq

        # Create the filter coefficients using Butterworth filter design
        b, a = signal.butter(order, normalized_cutoff_freq, btype='lowpass')

        zi1 = signal.lfiltic(b, a, 0)
        zi2 = signal.lfiltic(b, a, 0)
        zi3 = signal.lfiltic(b, a, 0)


        print("Filter Created")

        # Import desired tarjectory points
        trajectory_file_name = folder + "desired_path.csv"
        desired_trajectory = np.genfromtxt(trajectory_file_name, delimiter=',')

        num_rows, num_cols = desired_trajectory.shape
        n = multiprocessing.Value('i',0);
        
        samples = 2

        WIDTH1 = wav_file1.getsampwidth()
        WIDTH2 = wav_file2.getsampwidth()
        WIDTH3 = wav_file3.getsampwidth()

        data1 = wav_file1.readframes(samples)
        data2 = wav_file2.readframes(samples)
        data3 = wav_file3.readframes(samples)

    ##    FACTOR1 = 1
    ##    FACTOR2 = 1
    ##    FACTOR3 = 1




        # Create and start the video capture process
        video_process = multiprocessing.Process(target=video_capture, args=(queue,))
        video_process.start()
        print("Video Started")

        # Create and start the image processing process
        image_process = multiprocessing.Process(target=image_processing, args=(queue,queue_tag_p,))
        image_process.start()
        print("Image Processing Started")

        # Wait for the processes to complete
    ##    video_process.join()
    ##    image_process.join()

        
        time.sleep(5)
        print("Voltage Supply Started")
        while (data1 and data2 and data3 and n.value < num_cols):

##            data_1_co, zi1 = butter_lowpass_filter(np.array([control_1.value]), zi1)
##            data_2_co, zi2 = butter_lowpass_filter(np.array([control_2.value]), zi2)
##            data_3_co, zi3 = butter_lowpass_filter(np.array([control_3.value]), zi3)
            
            data1 = audioop.mul(data1, WIDTH1, FACTOR1)
            data2 = audioop.mul(data2, WIDTH2, FACTOR2)
            data3 = audioop.mul(data3, WIDTH3, FACTOR3)

##            data1 = audioop.mul(data1, WIDTH1, data_1_co[0])
##            data2 = audioop.mul(data2, WIDTH2, data_2_co[0])
##            data3 = audioop.mul(data3, WIDTH3, data_3_co[0])
                        
            data_2 = np.frombuffer(data2, dtype=np.int16).reshape(-1,1)
            data_3 = np.frombuffer(data3, dtype=np.int16).reshape(-1,1)
            data_4 = np.concatenate((data_2,data_3), axis = 1)

            data2 = data_4.tobytes()

            stream1.write(data1)
            stream2.write(data2)

                            # Move the iterator forward
            data1 = wav_file1.readframes(samples)
            data2 = wav_file2.readframes(samples)
            data3 = wav_file3.readframes(samples)

            n.value += samples

        video_process.kill()
        image_process.kill()

        stream1.close()
        stream2.close()

        wav_file1.close()
        wav_file2.close()
        wav_file3.close()
        
        if queue_tag_p.empty():
            Factor_index -= Factor_index
            print(f"No data")
        else:
            path_iter = queue_tag_p.get()
        
            path_iter = np.concatenate((path_iter, np.array([[FACTOR1*max_output],[FACTOR2*max_output],[FACTOR3*max_output]])), axis = 0)
            
            path = np.concatenate((path,path_iter), axis = 1)
            
            np.savetxt("position_voltage3.csv", path, delimiter=",")
            factor_counter+=1



        queue.empty()
        queue.close()
        queue_tag_p.empty()
        queue_tag_p.close()
        trigger.release
        time.sleep(1)
        
        print(f"Playing stops")

        

        print(f"Process: {factor_counter/Factor_cols*100}%")

    cv2.destroyAllWindows()


