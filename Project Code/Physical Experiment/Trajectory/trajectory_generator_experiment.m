%% Initialization
clear; 
close all; 
clc;

%% basic parameters
fs = 8000;              % sampling rate
v = 0.5;                % velocity
dt = 1/fs;              % time interval

%% circle trajectory
% r = 0.5;                % radius
% T_circle = 2*pi*r/v;               % toal time for one loop
% N_circle = round(T_circle*fs);     % total sampling points
% theta = linspace(0, 2*pi, N_circle); % polar angle
% 
% x_start = linspace(0, r, round(r/v*fs));
% z_start = zeros(size(x_start));
% y_start = zeros(size(x_start));
% 
% hold_steps = 400;
% x_hold = r * ones(1, hold_steps);
% y_hold = zeros(1, hold_steps);
% z_hold = zeros(1, hold_steps);
% 
% x = [x_start, x_hold];
% y = [y_start, y_hold];
% z = [z_start, z_hold];
% 
% x_circle = r * cos(theta);
% y_circle = r * sin(theta);
% z_circle = zeros(1, N_circle);
% 
% x_return = linspace(r, 0, round(r/v*fs));
% y_return = zeros(size(x_return));
% z_return = zeros(size(x_return));
% 
% for i = 1:5
%     x = [x, x_circle];
%     y = [y, y_circle];
%     z = [z, z_circle];
% end
% 
% x = [x, x_hold, x_return];
% y = [y, y_hold, y_return];
% z = [z, z_hold, z_return];


%% name trajectory
% points = [0.1,0,0.3; -0.1,0,0.2; 0,0,0.25; 0,0,0.15; -0.2,0,0.15; 0.2,0,0.15; 0,0,0.15; -0.3,0,-0.15;
%     -0.1,0,0.05; -0.1,0,-0.2; -0.1,0,0.05; 0,0,0.15; 0.3,0,-0.15; 0.1,0,0.05; 0.1,0,-0.35];
% 
% x = []; 
% y = []; 
% z = [];
% 
% for i = 1:size(points)-1
%     p_start = points(i,:);
%     p_end = points(i+1,:);
%     L = norm(p_end-p_start);
%     T = L/v;
%     N_name = T*fs;
% 
%     xi = linspace(p_start(1), p_end(1), N_name);
%     yi = linspace(p_start(2), p_end(2), N_name);
%     zi = linspace(p_start(3), p_end(3), N_name);
% 
%     x = [x, xi];
%     y = [y, yi];
%     z = [z, zi];
% end

% % Fz test trajectory
% x_start = 0;
% y_start = 0;
% z_start = 0;
% 
% hold_time = 5;
% x_hold = zeros(1, hold_time * fs);
% y_hold = zeros(1, hold_time * fs);
% z_hold = 0.5 * ones(1, hold_time * fs);
% 
% x = [x_start, x_hold]; 
% y = [y_start, y_hold]; 
% z = [z_start, z_hold]; 

% z_trajectory
n_cycles = 5;
z_points = zeros(1, n_cycles*2+1);
for i = 1:n_cycles
    z_points(2*i)   = -0.3;
    z_points(2*i+1) =  0.3;
end

x = []; y = []; z = [];

hold_time = 5;
x_hold = zeros(1, hold_time * fs);
y_hold = zeros(1, hold_time * fs);
z_hold = 0.5 * ones(1, hold_time * fs);

for i = 1:length(z_points)-1
    d = abs(z_points(i+1)-z_points(i));
    segment_time = d / v;
    N_segment = round(segment_time * fs);
    zi = linspace(z_points(i), z_points(i+1), N_segment);
    xi = zeros(1, N_segment);
    yi = zeros(1, N_segment);   
    x = [x, xi, x_hold];
    y = [y, yi, y_hold];
    z = [z, zi, z_hold];
end

%% whole trjectory
fz = ones(1, length(x)) * 0.015;
% circuleposition = [x; y; z; fz];
circuleposition = [x; y; z];
stop = zeros(0.05*fs,1);

%% plotting
figure
plot3(x, y, z, 'b'); axis equal
xlabel('x'); ylabel('y'); zlabel('z');
title('Circle Trajectory')
xlim([-0.6 0.6]);
ylim([-0.6 0.6]);
zlim([-0.6 0.6]);

%% output date and audio files
%folder_name = "Circle_stop";
%folder_name = "Circle";
folder_name = "z_direction";

folder = folder_name + "/";
mkdir(folder_name);

writematrix(circuleposition, folder+"desired_path.csv");
writematrix(stop,folder+"stop.csv");

data1 = ones(length(x), 1);
data2 = ones(length(x), 2);
audiowrite(folder+"control01.wav", data1, fs);
audiowrite(folder+"control02.wav", data2, fs);

%% save
%csvwrite('circle_reference_voltage.csv', U_ref');

% write signal .wav：
% audiowrite(folder+"OP_1.wav", voltage(1,:)', fs);
% audiowrite(folder+"OP_2.wav", voltage(2,:)', fs);
% audiowrite(folder+"OP_3.wav", voltage(3,:)', fs);