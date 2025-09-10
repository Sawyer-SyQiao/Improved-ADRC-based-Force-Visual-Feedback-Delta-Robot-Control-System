%% Initialization
clear; close all; clc;

%% Basic parameters
fs  = 8000;           % sampling rate (Hz)
dt  = 1/fs;           % time interval
A   = 0.40;           % step amplitude (units must match your NNF training, e.g., mm)
t_pre   = 0.50;       % pre-step hold (s)
t_hold  = 1.50;       % hold after step (s)
t_gap   = 1.00;       % gap/return-to-zero hold (s)

% Convert to samples
N_pre  = round(t_pre  * fs);
N_hold = round(t_hold * fs);
N_gap  = round(t_gap  * fs);

%% Helper to build one axis step block:  +A -> 0 -> -A -> 0
mk_block = @(Aamp) [ ...
    zeros(1, N_pre), ...
    Aamp * ones(1, N_hold), ...
    zeros(1, N_gap), ...
    (-Aamp) * ones(1, N_hold), ...
    zeros(1, N_gap) ];

%% Build sequences axis by axis (other axes = 0 during a given axis test)
% X-axis block
x_blk = mk_block(A);
y_blk = zeros(size(x_blk));
z_blk = zeros(size(x_blk));

% Y-axis block
x_blk2 = zeros(size(x_blk));
y_blk2 = mk_block(A);
z_blk2 = zeros(size(x_blk));

% Z-axis block
x_blk3 = zeros(size(x_blk));
y_blk3 = zeros(size(x_blk));
z_blk3 = mk_block(A);

% Concatenate: X test -> Y test -> Z test
x = [x_blk,  x_blk2,  x_blk3];
y = [y_blk,  y_blk2,  y_blk3];
z = [z_blk,  z_blk2,  z_blk3];

% Optional: initial & final zeros to settle (uncomment if needed)
% N_settle = round(0.5*fs);
% x = [zeros(1,N_settle), x, zeros(1,N_settle)];
% y = [zeros(1,N_settle), y, zeros(1,N_settle)];
% z = [zeros(1,N_settle), z, zeros(1,N_settle)];

%% Fz channel (all zeros for b0 estimation)
fz = zeros(1, length(x));

%% Pack and export
desired_path = [x; y; z; fz];

folder_name = "Step_b0";
folder = folder_name + "/";
if ~exist(folder_name, 'dir'); mkdir(folder_name); end

writematrix(desired_path, folder + "desired_path.csv");
writematrix([],           folder + "stop.csv");   

% Audio placeholders
data1 = ones(length(x), 1);   % 1-channel
data2 = ones(length(x), 2);   % 2-channel
audiowrite(folder + "control01.wav", data1, fs);
audiowrite(folder + "control02.wav", data2, fs);

%% Quick visualization
figure('Name','Step sequences for b0 estimation');
t = (0:length(x)-1)/fs;
subplot(3,1,1); plot(t, x); grid on; ylabel('x'); title('Axis steps');
subplot(3,1,2); plot(t, y); grid on; ylabel('y');
subplot(3,1,3); plot(t, z); grid on; ylabel('z'); xlabel('Time (s)');
