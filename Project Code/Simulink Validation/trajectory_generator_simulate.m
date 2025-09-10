%% Initialization
clear; 
close all; 
clc;

%% basic parameters
fs = 8000;              % sampling rate
v = 1.0;                % velocity
dt = 1/fs;              % time interval

% circle trajectory
r = 0.4;                % radius
T_circle = 2*pi*r/v;               % toal time for one loop
N_circle = round(T_circle*fs);     % total sampling points
theta = linspace(0, 2*pi, N_circle); % polar angle

x_start = linspace(0, r, round(r/v*fs));
z_start = zeros(size(x_start));
y_start = zeros(size(x_start));

x_circle = r * cos(theta);
z_circle = zeros(1, N_circle);
y_circle = r * sin(theta);

x_return = linspace(r, 0, round(r/v*fs));
y_return = zeros(size(x_return));
z_return = zeros(size(x_return));

x = x_start; y = y_start; z = z_start;
for i = 1:5
    x = [x, x_circle];
    y = [y, y_circle];
    z = [z, z_circle];
end
x = [x, x_return]; y = [y, y_return]; z = [z, z_return];

fz = ones(1, length(x))*2*1e-2;

%% whole trajectory + z-direction force
x_ref = [x; y; z];
t = (0:(length(x)-1))'/fs;
%% plot reference trajectory
figure
subplot(1,2,1)
plot3(x, y, z, 'b'); axis equal
xlabel('x'); ylabel('y'); zlabel('z');
title('Reference Trajectory')
xlim([-0.6 0.6]); ylim([-0.6 0.6]); zlim([-0.6 0.6]);

subplot(1,2,2)
plot(t, x_ref(1,:)); hold on,
plot(t, x_ref(2,:));
plot(t, x_ref(3,:)); hold off
title('Reference Position');
legend('X','Y','Z');

%% export for Simulink
ref_signals = x_ref';
ref_time = t;

ref_struct.time = ref_time;
ref_struct.signals.values = ref_signals;
ref_struct.signals.dimensions = 3;

%%
load('G0.mat');
A = LinearAnalysisToolProject.LocalVariables.Value.A;
B = LinearAnalysisToolProject.LocalVariables.Value.B;
C = LinearAnalysisToolProject.LocalVariables.Value.C;
D = LinearAnalysisToolProject.LocalVariables.Value.D;