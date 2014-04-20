
function main

addpath('./util')

evaluate_eye = evaluate();

%% Load dataset
if 1
    kPath='./dataset/';
    [Sequences, Calib_Brick, display_param] = load_sequences(kPath);
else
   % [Sequences, Calib_Brick, display_param] = load_artificial;
end


%% Run the experiment
evaluate_eye(Sequences, Calib_Brick, display_param)
             
return

end


%% Load whole datasets for the experiment
function [Sequences, Calib_Brick, display_param] = load_sequences(kPath)

Sequences{1}=cell(1,1);
SeqStr = {''};

%%% sequence1
s_idx=1;
b_idx=1;
kSubPath=strcat(SeqStr{s_idx},'1\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'2\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'3\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'4\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'5\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'6\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'7\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'8\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'9\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;
kSubPath=strcat(SeqStr{s_idx},'10\');
Sequences{s_idx}{b_idx}=load_sub2(kPath, kSubPath); b_idx=b_idx+1;

if 1 %% Visualize 3D GT points
xyz=[];
xyz_all=[];
figure(222);clf;hold on;grid on;axis equal;
n = b_idx-1;
color_map = lines(n);
draw_axis(eye(3),zeros(3,1),0.1,eye(3),2);
for k=1:n
    xyz = [ Sequences{s_idx}{k}.xyz_gt];
    xyz_all = [ xyz_all xyz ];
    x=xyz(1,:);
    y=xyz(2,:);
    z=xyz(3,:);
    plot3(x,y,z,'.','Color',color_map(k,:),'MarkerSize',15);
end
x=xyz_all(1,:);
y=xyz_all(2,:);
z=xyz_all(3,:);
xmin=min(x);
xmax=max(x);
ymin=min(y);
ymax=max(y);
zmin=min(z);
zmax=max(z);

%xrange=xmax-xmin
%xmean=mean(x)
%yrange=ymax-ymin
%ymean=mean(y)
%zrange=zmax-zmin
%zmean=mean(z)

xlim([xmin,xmax]);
ylim([ymin,ymax]);
zlim([zmin,max([zmax 0])]);

% view angles
az =    130;
el =     12;
view(az,el);
title('3D points collected during SPAAM')
% rotate camera
camorbit(-90,0,'data',[1 0 0]);
camorbit(-100,0,'data',[0 1 0]);
figure(1);clf;hold on;grid on;axis equal;

end

kSubPath='Base\';
Calib_Brick=load_sub2(kPath, kSubPath);

[display_param.R_WT, display_param.t_WT, display_param.q_WT] = loadUbitrackPose0(strcat(kPath,'PoseFromEyeCam2WorldCam.txt'));

display_param.q_WT = mat2ubitrackQuat(display_param.R_WT);

display_param.t_WS0_z = 7.80000000000001e-01;

dir = '.\';
display_param_nonlin = load(strcat(kPath,'display_param_nonlinear'));
display_param_linear = load(strcat(kPath,'display_param_linear'   ));

if 1 % no artificial noise
    display_param_nonlin.R_WS(2:3,:) = -display_param_nonlin.R_WS(2:3,:);
    display_param_nonlin.t_WS(2:3)   = -display_param_nonlin.t_WS(2:3);
    display_param_linear.R_WS(2:3,:) = -display_param_linear.R_WS(2:3,:);
    display_param_linear.t_WS(2:3)   = -display_param_linear.t_WS(2:3);
end

% estimates by non-linear optimization
display_param.distortion        = display_param_nonlin.param;
display_param.t_WS0_z           = display_param_nonlin.t_WS(3);
display_param.t_WS  = display_param_nonlin.t_WS;
display_param.R_WS  = display_param_nonlin.R_WS;
display_param.alpha = display_param_nonlin.alpha;

% estimates by linear optimization
display_param.distortion_linear = display_param_linear.param;
display_param.t_WS0_z_linear    = display_param_linear.t_WS(3);
display_param.t_WS_linear       = display_param_linear.t_WS;
display_param.R_WS_linear       = display_param_linear.R_WS;
display_param.alpha_linear      = display_param_linear.alpha;

display_param.w = display_param_linear.w;
display_param.h = display_param_linear.h;

visualize_calib(display_param)

end

%% Load a block data
% A block contains the following
% 
function block = load_sub2(kPath, kSubPath)

kDataDir = strcat(kPath, kSubPath);
% GT data
[xy0,xyz0] = load2D3D(kPath, kSubPath);
block.xy_gt = xy0;
block.xyz_gt = xyz0;

% SPAAM data
Data = loadData2(kDataDir);
block.P_WE0 = Data.P_gt;
block.R_WE0 = Data.R_WE;
block.t_WE0 = Data.t_WE;
block.K_E0  = Data.K;
block.cfg_gt    = 'k';
block.cfg_eye   = 'm';
block.cfg_SPAAM = 'c';

% Raw eye position data (not disambiguated)
block.t_ETs = load(strcat(kDataDir,'eye_positions.txt'))';
[block.t_ET_mean, block.t_ET_mean2] = eye_point_mean(block.t_ETs);

%% A local function for loading SPAAM datasets
    function out = loadData2(dir)
        strcat(dir,'SPAAM_extrinsic.calib')
        [out.R_WE, out.t_WE, out.q_WE] = loadUbitrackPose0(strcat(dir,'SPAAM_extrinsic.calib'));
        out.P_gt = openUbitrack3x4MatrixCalib(strcat(dir,'SPAAM_calibration.calib'));
        out.K    = openUbitrack3x3MatrixCalib(strcat(dir,'SPAAM_intrinsic.calib'));
        % Convert a quaternion to a rotation matrix
        out.R_WE = ubitrackQuat2Mat(out.q_WE);
        return
    end
%% A local function for loading 2D/3D point data sets
    function [xy,xyz] = load2D3D(kPath, kSubPath)
        kDataDir = strcat(kPath,kSubPath);
        xyz = load(strcat(kDataDir,'MarkerSPAAM_DataSPAAM3D.txt'))'; %%% SPAAM 3D
        xy  = load(strcat(kDataDir,'MarkerSPAAM_DataSPAAM2D.txt'))'; %%% SPAAM 2D
        xy=round(xy);
    end
end

%% Load ubitrack-style 6-DoF measurement
function [R, t, q]  = loadUbitrackPose0(file)
tmp=openUbitrack6DPoseCalib(file); % [ timestamp [quarternion] position ]
%time_stamp =tmp(1);
q = tmp(2:5); 
R = ubitrackQuat2Mat(q);
t = tmp(end-2:end);
end

%% Disambiguate eye positions
% this method is based on k-means under the assumption that the
% eye positions stay similar for a certain duration.
function [mean_eye, mean_eye2]= eye_point_mean(t_CEs)

% K-means
X=t_CEs';
% repeat k-means
N =10;%100
mean_eyes = zeros(3,N);
mean_eye2 = median(X)';

for k = 1:N
    K = 2;
    [idx,ctrs] = kmeans(X,K,...
        'Distance','city',...
        'Replicates',5);
    idx_odd=idx(1:2:end);
    idx_even=idx(2:2:end);
    
    % Two candidates from a same image can not be in the same cluster!
    % This might cause empty clusters, though (this is checked later on)
    idx_duplicated = idx_odd~=idx_even;
    idx_duplicated = true(size(idx_duplicated));
    idx(1:2:end) =  idx_odd.*idx_duplicated;
    idx(2:2:end) =  idx_even.*idx_duplicated;
    X1=X(idx==1,:);
    X2=X(idx==2,:);
    v1=mean(var(X1));
    v2=mean(var(X2));
    if v1<v2
        mean_eye = mean(X1,1);
    else
        mean_eye = mean(X2,1);
    end
    if v1<v2
        median_eye = median(X1,1);
    else
        median_eye = median(X2,1);
    end
    
    mean_eyes(:,k) = mean_eye';
    mean_eyes(:,k) = median_eye';
end

%%% remove NaN vectors
valid_index = find( isnan (mean_eyes(1,:))==false );
mean_eye = mean(mean_eyes(:,valid_index),2);
%mean_eye = mean_eye2;
end

%% For debugging, visualize calibration misc. calib. parameters
function visualize_calib(display_param)
len = 0.02;

R_WT = display_param.R_WT;
t_WT = display_param.t_WT;
R_TW=R_WT';
t_TW=-(R_WT')*t_WT;

R_WS = display_param.R_WS;
t_WS = display_param.t_WS;
R_SW=R_WS';
t_SW=-(R_WS')*t_WS;

fig_id = 43;
figure(fig_id);set(gcf,'visible','off'); clf; hold on; grid on; axis equal;
draw_axis(eye(3),zeros(3,1),len);
draw_axis(R_TW,t_TW,len,cool(3) );
%draw_axis(R_SW,t_SW,len);

fig_id = 44;
len=0.04;
figure(fig_id);set(gcf,'visible','off'); clf; hold on; grid on; axis equal;
draw_axis(R_TW,t_TW,len,eye(3),3 );
draw_axis(eye(3),zeros(3,1),len,eye(3),3 );
title('3D eye position estimates (w/ tracker and world cam.)');
figure(1);set(gcf,'visible','off'); clf;
end

%% Convert "Print Sink (Quaternion)" output format to a rotation matrix
function R = ubitrackQuat2Mat(q)
if(size(q,1) == 1)
    q=q';
end
I=[0 0 -1 0; 0 -1 0 0 ; 1 0 0 0; 0 0 0 1];
q;
R = diag([-1 -1 1])*quat2dcm( (I*q)')';
end

%% Convert  a rotation matrix to "Print Sink (Quaternion)" output format
function q = mat2ubitrackQuat(R)
q0=dcm2quat(R);
q=[q0(2:4) -q0(1)];
end
