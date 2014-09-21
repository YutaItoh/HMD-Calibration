function fhandle = evaluate
fhandle = @evaluate_;
end

%% Run the experiment
function evaluate_ (Sequences, Calib_Brick, display_param)

Font = 'CMU Serif';
FontSize = 14;
% Change default axes fonts.
set(0,'DefaultAxesFontName', Font)
set(0,'DefaultAxesFontSize', FontSize)
set(0,'DefaultAxesFontWeight', 'bold')

% Change default text fonts.
set(0,'DefaultTextFontname', Font)
set(0,'DefaultTextFontSize', FontSize)
set(0,'DefaultTextFontWeight', 'bold')

%% Initialize variables for plottinf results
SequenceNum = length(Sequences);
BlockNum    = length(Sequences{1});
plot_cfg.SequenceNum = SequenceNum;
plot_cfg.BlockNum = BlockNum;
plot_cfg.Calib_Brick = Calib_Brick;
plot_cfg.display_param = display_param;

% draw flag
DRAW_FLAG = true;%false;
set(gcf,'visible','off')
plot_cfg.DRAW_FLAG = DRAW_FLAG;

% Create "pointer" variables
plot_cfg.pt_t_EWs = Pointer([]);
plot_cfg.pt_2D_est_xy   = Pointer([]);
plot_cfg.pt_2D_gt_xy    = Pointer([]);
plot_cfg.pt_3D_gt_xyz   = Pointer([]);
plot_cfg.pt_2D_error_xy = Pointer([]);
plot_cfg.pt_2D_error_r_theta = Pointer([]);
plot_cfg.pt_undistort_GT = Pointer([]);
plot_cfg.pt_corr_GT_uv_vs_ErrorVector = Pointer([]);
plot_cfg.pt_corr_GT_u_vs_errror_r = Pointer([]);
plot_cfg.pt_corr_GT_v_vs_errror_r = Pointer([]);
plot_cfg.pt_corr_GT_x_vs_errror_r = Pointer([]);
plot_cfg.pt_corr_GT_y_vs_errror_r = Pointer([]);
plot_cfg.pt_corr_GT_z_vs_errror_r = Pointer([]);
plot_cfg.pt_corr_GT_u_vs_errror_t = Pointer([]);
plot_cfg.pt_corr_GT_v_vs_errror_t = Pointer([]);
plot_cfg.pt_corr_GT_x_vs_errror_t = Pointer([]);
plot_cfg.pt_corr_GT_y_vs_errror_t = Pointer([]);
plot_cfg.pt_corr_GT_z_vs_errror_t = Pointer([]);

%% Create experiment conditions %%
function out = create_condition( name, instance, plot_color, plot_marker_shape, plot_axis_color)
    out.name = name;
    out.instance = instance; % a function handle
    out.plot_color = plot_color;
    out.plot_marker_shape = plot_marker_shape;
    out.plot_axis_color = plot_axis_color;
end
id = 1; % do not forget to increment this;

methodNameSAPAAM        = 'SPAAM';
methodNameDegradedSPAAM = 'Deg.SPAAM';
methodNameRecycle       = 'Recycle';
methodNameRecycleManual = 'Recycle Setup (Manual)';
methodNameFull          = 'Full';

% SPAAM (Training condition)
condition{id} = create_condition(methodNameSAPAAM,        @compare_by_SPAAM,         'b', '.', eye(3) );id=id+1;
% Degraded SPAAM (Testing confition)
condition{id} = create_condition(methodNameDegradedSPAAM, @compare_by_DegradedSPAAM, 'c', '.', eye(3) );id=id+1;
% Eye calibration option 2 (t_WS_z is manual value)
condition{id} = create_condition(methodNameRecycle ,      @compare_by_INDICA_Recycle,        'r', '.', cool(3) );id=id+1;
% Eye calibration option 1 (Full display param)
condition{id} = create_condition(methodNameFull,          @compare_by_INDICA_Full,        'm', '.', cool(3) );id=id+1;
num_of_conditions = length(condition);

%% Set up figure configurations
plot_cfg.method_names = cell(1,0);
plot_cfg.fig2D    = 1; % 2D projection error bias analysis
plot_cfg.fig2D2   = 2; % 2D projection error bias analysis
plot_cfg.fig2D3   = 3; % 2D projection error bias analysis 2
plot_cfg.fig3D    = 4; % 3D eye position analysis
plot_cfg.fig3D2   = 5; % 3D eye position analysis 2
plot_cfg.figError = 6;
plot_cfg.figCorrelationError = 7;
plot_cfg.figHisto = 8;
figure(plot_cfg.fig2D); clf; hold on; grid on; axis equal;
plot_cfg.subgfig_row = 1; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% also compute distorted case
plot_cfg.subgfig_col = num_of_conditions;
plot_cfg.subgfig_range = 20;
plot_cfg.subgfig_idx = 1;
% 2D projection error bias analysis
figure(plot_cfg.fig2D2); clf; hold on; grid on; axis equal;
for k=1:plot_cfg.subgfig_row*plot_cfg.subgfig_col
    subplot(plot_cfg.subgfig_row,plot_cfg.subgfig_col,k);
    clf;
    hold on;
    grid on;
    axis equal;
end
% 2D projection error bias analysis 2
figure(plot_cfg.fig2D3); clf;
for k=1:plot_cfg.subgfig_row*plot_cfg.subgfig_col
    subplot(plot_cfg.subgfig_row,plot_cfg.subgfig_col,k);
    clf;
    hold on;
    grid on;
end
% 3D eye position analysis
figure(plot_cfg.fig3D);  clf;
for k=1:plot_cfg.SequenceNum
    subplot(plot_cfg.SequenceNum,1,k);
    clf;
    hold on;
    grid on;
    
end
% 3D eye position analysis 2
figure(plot_cfg.fig3D2); clf;
for k=1:plot_cfg.subgfig_row*plot_cfg.subgfig_col
    subplot(plot_cfg.subgfig_row,plot_cfg.subgfig_col,k);
    clf;
    hold on;
    grid on;
end

%% %%%%%%%%%%%%%%%%%%
% Run the experiments %%
%%%%%%%%%%%%%%%%%%%%%%%%
% variables used for box plot, store all error samples with their method labels
error_sets           = [];
error_sets_labels    = [];
error_sets_label_idx = 1;
plot_cfg.consider_distortion = false;
plot_cfg.plot2D_handle = [];
for j=1:plot_cfg.subgfig_row
    if j==2
        plot_cfg.consider_distortion = true;
    end
        
    %% Recompute SPAAM by DLT
    SequenceNum = length(Sequences);
    BlockNum    = length(Sequences{1});
    for s_idx = 1:SequenceNum
        for k = 1:BlockNum
            block_gt = Sequences{s_idx}{k};
            xyz = block_gt.xyz_gt;
            uv = block_gt.xy_gt;
            %keyboard
           %% Take radial distortion into account
            if plot_cfg.consider_distortion
                dist=plot_cfg.display_param.distortion;
                cx=dist.cx;
                cy=dist.cy;
                k1=dist.k1;
                k2=dist.k2;
                k3=dist.k3;
                p1=dist.p1;
                p2=dist.p2;
                uv = distort(uv,cx,cy,k1,k2,k3,p1,p2);
            end
            P_WE0 = DLT(uv,xyz);
            %[P_WE0, rperr] = RefinePMat(uv, xyz, P_WE0);
            [K_E0, R_WE0, t_WE0] = decompose_projection(P_WE0);
            K_E0 = K_E0/abs(K_E0(3,3));
            block_gt.P_WE0 = P_WE0;
            block_gt.t_WE0 = t_WE0;
            block_gt.R_WE0 = R_WE0;
            block_gt.K_E0  = K_E0;
        end
    end
    %% Evaluate each method
    for k = 1:num_of_conditions
        initialize_plot_variables(plot_cfg);
        name = condition{k}.name;
        fun_method = condition{k}.instance;
        plot_cfg.method_names = [ plot_cfg.method_names, name ];
        plot_cfg.color = condition{k}.plot_color;
        plot_cfg.shape = condition{k}.plot_marker_shape;
        plot_cfg.axis_color =  condition{k}.plot_axis_color;
        plot_cfg.is_estimated_z_used = true;
        if strcmp(name, methodNameRecycleManual)
            plot_cfg.is_estimated_z_used = false;
        end
        
        display(strcat('Running: ',name));
        errors = fun_method(Sequences,plot_cfg);
        plot_cfg.error_sets_label_idx = error_sets_label_idx;
        [ plot2D_handle ] = plot_2D_error_xy(plot_cfg);
        plot_cfg.plot2D_handle{k} = plot2D_handle;
        error_sets           = [ error_sets; errors];
        error_sets_labels    = [ error_sets_labels; error_sets_label_idx*ones(size(errors,1),1)];
        error_sets_label_idx = error_sets_label_idx +1;
        plot_cfg.subgfig_idx = plot_cfg.subgfig_idx + 1;
        
    end
end
error_sets_label_idx=error_sets_label_idx-1;

%% Boxplot of the 2D projection error
num_corr = 1;%<=11
figure(plot_cfg.figError); clf; hold on; grid on; %axis equal;
%fig_width =1200;
%fig_height=1500;
%screen_size=get(gcf,'Position');
%set(gcf,'Position',[screen_size(1) screen_size(2) fig_width fig_height]);
subplot(num_corr+1,1,1);hold on;grid on; 

boxplot(error_sets, error_sets_labels, ...
    ...%'labels',{'SPAAM','Deg. SPAAM', 'Option 2', 'Op.2 t_WS_z given'
    ...%,'Option 1'});
    'labels',plot_cfg.method_names);
%xticklabel_rotate([1:length(plot_cfg.method_names)],45,plot_cfg.method_names,'interpreter','none');
title('Reprojection errors')
ylabel('MSD [pixel]')
e_max=max( error_sets );
ylim([0 e_max*1.05+eps]);

if DRAW_FLAG == false
    return;
end

%% Plot the correlation of the 2D projection error 
for k = 1:num_corr
    corr_title = [];
    error_correlations = [];
    switch k
        case 1
            error_correlations  = plot_cfg.pt_corr_GT_uv_vs_ErrorVector.get();
            corr_title = 'GT 2D postions / error vectors';
    end
    subplot(num_corr+1,1,1+k);hold on; grid on; 
    hb=bar(error_correlations);
    set(gca, 'xtick',1:1:length(plot_cfg.method_names));
    if k == num_corr
        set(gca,'xticklabel', plot_cfg.method_names);
    else
        set(gca,'xticklabel', cell(1,length(plot_cfg.method_names)));
    end
    title(corr_title );
    %ylabel('Corr. betw. Positions and Dirctions');
    e_max=max( abs(error_correlations) );
    ylim([-e_max e_max]*1.05);
    ylabel('Correlation');
    xlim([0.5 length(plot_cfg.method_names)+0.5]);
    % paint shaded color on the bars
    colormap jet;
    set(get(hb,'children'),'cdata', error_correlations);
    caxis([-e_max e_max]);
end


%% Plot a histogram of the reprojection errore
figure(plot_cfg.figHisto);
set(gcf,'visible','off')
clf;
emax=25;
for k = 1:num_of_conditions
    %subplot( 2, ceil(length(plot_cfg.method_names)/2), k);
    subplot( length(plot_cfg.method_names),1, k);
    data = error_sets(error_sets_labels==k);
    nbins = floor(length(data)/2);
    [f2,x2]=hist(data,nbins);
    hold on;grid on;
    scale_hist2density = trapz(x2,f2);
    bar(x2,f2);
    [f,xi] = ksdensity(data); 
    plot(xi,f*scale_hist2density,'r', 'LineWidth', 2); 
    xlim([0,emax]);
    title(plot_cfg.method_names(k));
    if(k==num_of_conditions)
        xlabel('MSD to GT points [pixel]')
        ylabel('Count/Scaled Density')
        legend('Count','Density')
    end
end


%% Tune figures
figure(plot_cfg.fig3D);
%draw_world( display_param ) 
for k=1:plot_cfg.SequenceNum
    subplot(plot_cfg.SequenceNum,1,k);
    hold on;
    axis equal;
end

figure(plot_cfg.fig2D);set(gcf,'visible','on');
%condition{1}.name

legend_names = cell(1,num_of_conditions);
legend_handles = zeros(1,num_of_conditions);
for k = 1:num_of_conditions
    legend_handles(k) = plot_cfg.plot2D_handle{k};
    legend_names{k}   = condition{k}.name;
end
legend(legend_handles,legend_names);

title('2D point visualizatoin')
xlabel('x [pixel]')
ylabel('y [pixel]')
set(gcf,'visible','on');
figure(plot_cfg.fig2D2);set(gcf,'visible','on');
figure(plot_cfg.fig2D3);set(gcf,'visible','on');
%figure(plot_cfg.fig3D); set(gcf,'visible','on');
figure(plot_cfg.fig3D2);set(gcf,'visible','on');
figure(plot_cfg.figHisto);set(gcf,'visible','on');

figure(44);set(gcf,'visible','on');
camorbit(90,0,'data',[0 1 0]);

return
end %%% function evaluate_eye

%% Evaluation of SPAAM
function errors = compare_by_SPAAM(Sequences,plot_cfg)
    SequenceNum = length(Sequences);
    BlockNum    = length(Sequences{1});
    ErrorNum = SequenceNum*BlockNum;
    errors=zeros(ErrorNum,1);
    e_idx = 0;
    for s_idx = 1:SequenceNum
        plot_cfg.s_idx=s_idx;
        for k = 1:BlockNum
            block_spaam = Sequences{s_idx}{k};
            block_gt    = Sequences{s_idx}{k};
            
            param = get_SPAAM(block_spaam);
            e_idx = e_idx + 1;
            errors(e_idx,1) = analyze_estimate(param, block_gt, plot_cfg);
        end
    end
end

%% Evaluation of Degraded SPAAM
function errors = compare_by_DegradedSPAAM(Sequences,plot_cfg)
    SequenceNum = length(Sequences);
    BlockNum    = length(Sequences{1});
    ErrorNum = SequenceNum*(SequenceNum-1)*BlockNum*BlockNum;
    %errors=zeros(ErrorNum,1);
    e_idx = 0;
    for s_idx1 = 1:SequenceNum
        plot_cfg.s_idx=s_idx1;
        for s_idx2 = 1:SequenceNum
            for k = 1:BlockNum
                for j = 1:BlockNum
                    if k~=j || s_idx1~=s_idx2 % for all different sequence combinations
                        block_spaam = Sequences{s_idx1}{k};
                        block_gt    = Sequences{s_idx2}{j};

                        param = get_SPAAM(block_spaam);
                        e_idx = e_idx + 1;
                        errors(e_idx,1) = analyze_estimate(param, block_gt, plot_cfg);
                    end
                end
            end
        end
    end
end

%% Evaluation of Eye SPAAM Option 2 (the version with an old intrinsic param.)
function errors = compare_by_INDICA_Recycle(Sequences,plot_cfg)
    Calib_Brick = plot_cfg.Calib_Brick;
    display_param = plot_cfg.display_param;

    SequenceNum = length(Sequences);
    BlockNum    = length(Sequences{1});
    ErrorNum = SequenceNum*BlockNum;
    errors=zeros(ErrorNum,1);
    e_idx = 0;
    is_estimated_z_used = plot_cfg.is_estimated_z_used;
    for s_idx = 1:SequenceNum
        plot_cfg.s_idx=s_idx;
        for k = 1:BlockNum
            block_offline = Calib_Brick;
            block_online  = Sequences{s_idx}{k};
            block_gt      = block_online;
            param = get_INDICA_Recycle( block_online, block_offline, display_param, is_estimated_z_used );
            e_idx = e_idx + 1;
            errors(e_idx,1) = analyze_estimate(param, block_gt, plot_cfg);
        end
    end
end

%% Evaluation of Eye SPAAM Option 1 (the version with full display parameters.)
function errors = compare_by_INDICA_Full(Sequences,plot_cfg)
    Calib_Brick = plot_cfg.Calib_Brick;
    display_param = plot_cfg.display_param;

    SequenceNum = length(Sequences);
    BlockNum    = length(Sequences{1});
    ErrorNum = SequenceNum*BlockNum;
    errors=zeros(ErrorNum,1);
    e_idx = 0;
    for s_idx = 1:SequenceNum
        plot_cfg.s_idx=s_idx;
        for k = 1:BlockNum
            block_offline = Calib_Brick;%Sequences{s_idx}{k};
            block_online  = Sequences{s_idx}{k};
            block_gt      = block_online;
            param = get_INDICA_Full( block_online, block_offline, display_param );
            e_idx = e_idx + 1;
            errors(e_idx,1) = analyze_estimate(param, block_gt, plot_cfg);
        end
    end
end

%% Get Projection matrix from SPAAM
function out = get_SPAAM(block)
out.P_WE = block.P_WE0;
out.t_WE = block.t_WE0;
out.R_WE = block.R_WE0;

R_WE=out.R_WE;
t_WE=out.t_WE;

%% Draw Eye Ball (t_EW)
color=[1 0 0];
draw_eyeball(R_WE, t_WE, color);

end

function draw_eyeball(R_WE, t_WE, color)
fig_id = 44;
figure(fig_id);set(gcf,'visible','off');
t_EW=-R_WE'*t_WE;
len=0.05;
draw_axis(R_WE',t_EW,len);
[x0,y0,z0] = sphere(10);

nh=size(x0,1);
nw=size(x0,2);
xyz=[x0(:) y0(:) z0(:)]';
xyz=R_WE'*xyz;
x0=reshape(xyz(1,:),nh,nw);
y0=reshape(xyz(2,:),nh,nw);
z0=reshape(xyz(3,:),nh,nw);

hold on
radius = 12.60*0.001;
x = x0*radius+t_EW(1);
y = y0*radius+t_EW(2);
z = z0*radius+t_EW(3);
surf(x,y,z, 'FaceColor', color)  % sphere centered at t_EW
figure(22);set(gcf,'visible','off');%set random one
end

    
%% Get Projection matrix by using eye position and option 2 (Recycle)
function out = get_INDICA_Recycle( block_online, block_offline, display_param, is_estimated_z_used )
R_WS = block_offline.R_WE0;
t_WE0= block_offline.t_WE0;
K_E0 = block_offline.K_E0;
%keyboard
R_WT = display_param.R_WT;
t_WT = display_param.t_WT;
if is_estimated_z_used
    t_WS_z = display_param.t_WS0_z;%7.80000000000001e-01
else
    t_WS_z = 7.80000000000001e-01;
    %t_WS_z = 1.3;
end
t_ET = block_online.t_ET_mean;

% Flip the eye coordinate system
t_ET(3) = -abs(t_ET(3));

[P_WE,t_WE] = INDICA_Recycle(R_WS,R_WT,t_WT,t_ET,t_WS_z,K_E0,t_WE0);

out.P_WE = P_WE;
out.t_WE = t_WE;
out.R_WE = R_WS;

end



%% Get Projection matrix by using eye position and option 1 (Full display parameters)
function out = get_INDICA_Full( block_online, block_offline, display_param )
R_WT = display_param.R_WT;
t_WT = display_param.t_WT;
t_ET = block_online.t_ET_mean;

% Flip the eye coordinate system
t_ET(3) = -abs(t_ET(3));

%% Full display parameters are used
% if display param. are known...
% load 'display_param.mat' t_WS R_WS alpha
t_WS = display_param.t_WS;
R_WS  = display_param.R_WS;

w = display_param.w;
h = display_param.h;

alpha = display_param.alpha;
ax = alpha;
ay = alpha;

[P_WE,t_WE] = INDICA_Full(R_WS,R_WT,t_WT,t_ET,t_WS,ax,ay,w,h);

%% Draw Eye Ball (t_EW)
color=[0 1 0];
draw_eyeball(R_WS, t_WE, color);

%%
out.P_WE = P_WE;
out.t_WE = t_WE;
out.R_WE = R_WS;


end

%% Analyze a given estimate 
function error = analyze_estimate(param, block_gt, plot_cfg)
P_WE = param.P_WE;
t_WE = param.t_WE;
R_WE = param.R_WE;
if strcmp(plot_cfg.method_names(plot_cfg.subgfig_idx),'Partial(manual)')
    %P_WE_Option2=P_WE
end
xy_gt  = block_gt.xy_gt;
%%% Distort GT image points?
if plot_cfg.consider_distortion
    dist=plot_cfg.display_param.distortion;
    cx=dist.cx;
    cy=dist.cy;
    k1=dist.k1;
    k2=dist.k2;
    k3=dist.k3;
    p1=dist.p1;
    p2=dist.p2;
    xy_gt = distort(xy_gt,cx,cy,k1,k2,k3,p1,p2);
end
xyz_gt = block_gt.xyz_gt;

%% Reprojection Error
xy_est = projection(P_WE,xyz_gt);
error = xy1xy2diff(xy_gt,  xy_est); % dif. of projected 2D point sets
% out.xy_est = xy_est;
% out.error  = error;
if plot_cfg.DRAW_FLAG == false
    return;
end
% A local function for projecting 3D points xyz by P
    function xy=projection(P,xyz)
        xyzw=[xyz; ones(1,size(xyz,2))];
        xyz = P*xyzw;
        
        %xyz = P(1:3,1:3)*xyz+repmat(P(:,4),1,length(xyz));
        xy = [xyz(1,:)./xyz(3,:); xyz(2,:)./xyz(3,:)];
    end
% A local function for computing
% squared mean distance between 2 sets of vectors (their sizes must be DxN)
function error_xy = xy1xy2diff(xy1, xy2)
tmp= xy1-xy2;
error_xy = mean(sqrt(sum(tmp.*tmp)),2);
%error_xy = sum(sum(tmp.*tmp))/length(tmp)/3;
end

%% 2D projection error bias computation
xy_error = (xy_est-xy_gt);
r     = sqrt(sum(xy_error.*xy_error));
theta =acos(xy_error(1,:)./r);
theta(isnan(theta))=pi;
theta(xy_error(2,:)<0) = 2*pi-theta(xy_error(2,:)<0);

%plot_cfg.method_names(plot_cfg.subgfig_idx)
%xy_error
%r

% store errors into pointees
xy_est_all   = plot_cfg.pt_2D_est_xy.get();
xy_gt_all    = plot_cfg.pt_2D_gt_xy.get();
xyz_gt_all    = plot_cfg.pt_3D_gt_xyz.get();
error_xy     = plot_cfg.pt_2D_error_xy.get();
error_r_thet = plot_cfg.pt_2D_error_r_theta.get();
plot_cfg.pt_2D_est_xy.set([xy_est_all xy_est]);
plot_cfg.pt_2D_gt_xy.set( [xy_gt_all  xy_gt]); % store 2D error vector in the global variable
xyz_gt_E=[R_WE t_WE]*[xyz_gt;ones(1,size(xyz_gt,2))];
plot_cfg.pt_3D_gt_xyz.set([xyz_gt_all xyz_gt_E]); % 3D points
plot_cfg.pt_2D_error_xy.set([error_xy     xy_error]); % store 2D error vector in the global variable
plot_cfg.pt_2D_error_r_theta.set( [error_r_thet [r;theta] ]); % store 2D error vector in the global variable

% plot errors in the polar coordinates
% figure(plot_cfg.fig2D2);set(gcf,'visible','off')
% mean_xy = mean(xy_est,2);
% subplot(plot_cfg.subgfig_row,plot_cfg.subgfig_col,plot_cfg.subgfig_idx);
% hold on;grid on;
% plot(mean_xy(1),mean_xy(2),'ko');

%% 3D visualization in the WORLD coordinate system
figure(plot_cfg.fig3D);
set(gcf,'visible','off')
% fig_width=400;
% fig_height=500;
% screen_size=get(gcf,'Position');
% set(gcf,'Position',[screen_size(1) screen_size(2) fig_width fig_height]);
% s_num = plot_cfg.SequenceNum;
% subplot(s_num,1,plot_cfg.s_idx);
% hold on;
% grid on;
% 
% % Axis length
% axis_len = 0.05;
% %span_edges(-t_WE,xyz,strcat(cfg_gt,':'));
% draw_axis(R_WE',-R_WE'*t_WE, axis_len, plot_cfg.axis_color,2);% SPAAM eye
% title(strcat('Seq. ',num2str(plot_cfg.s_idx) ));
 
t_EW=-R_WE'*t_WE;
t_EWs_cell = plot_cfg.pt_t_EWs.get();
t_EWs_cell{plot_cfg.s_idx} = [t_EWs_cell{plot_cfg.s_idx} t_EW];
plot_cfg.pt_t_EWs.set(t_EWs_cell); % store 2D error vector in the global variable

end

%% Draw lines between 2 sets of 2D vectors (their sizes must be DxN)
function draw_edges(s1,s2,cfg)
D=size(s1,1);
N=size(s1,2);
p=NaN(D,3*N);
p(:,1:3:end)=s1;
p(:,2:3:end)=s2;
h=line(p(1,:),p(2,:), 'Color',cfg(1));
return
for k=1:size(s1,2)
    tmp=[s1(:,k) s2(:,k)];
    plot(tmp(1,:),tmp(2,:),cfg);
end
end

%% Initialize variables for plotting results
function initialize_plot_variables(plot_cfg)
    plot_cfg.pt_t_EWs.set(cell(1,plot_cfg.SequenceNum));
    plot_cfg.pt_2D_est_xy.set([]);
    plot_cfg.pt_2D_gt_xy.set([]);
    plot_cfg.pt_3D_gt_xyz.set([]);
    plot_cfg.pt_2D_error_xy.set([]);
    plot_cfg.pt_2D_error_r_theta.set([]);
end

%% Analyse Projected points
function [ plot2D_handle ] = plot_2D_error_xy(plot_cfg)
    xy_est = plot_cfg.pt_2D_est_xy.get();
    xy_gt  = plot_cfg.pt_2D_gt_xy.get();
    xyz_gt  = plot_cfg.pt_3D_gt_xyz.get();
    error_xy = plot_cfg.pt_2D_error_xy.get();
    error_r_theta = plot_cfg.pt_2D_error_r_theta.get();
    eye_xyz_cell = plot_cfg.pt_t_EWs.get();
    
    
    %% 2D visualization
    figure(plot_cfg.fig2D);
    set(gcf,'visible','off')
    fig_width=600;
    fig_height=500;
    screen_size=get(gcf,'Position');
    set(gcf,'Position',[screen_size(1) screen_size(2) fig_width fig_height]);
    EmpFactor = 1;
    xy_est2 = xy_est + (EmpFactor-1)*(xy_est-xy_gt);

    cfg_est=strcat(plot_cfg.color,plot_cfg.shape);
    plot2D_handle = plot( xy_est2(1,:), xy_est2(2,:), cfg_est);
    cfg_gt='k+';
    plot( xy_gt(1,:), xy_gt(2,:), cfg_gt);
    cfg_est=strcat(plot_cfg.color,'-');
    draw_edges(xy_gt,xy_est2,cfg_est)
    
    %% bias in x-y coordinate
    fig_width = 1500;
    fig_height =  500;
    figure(plot_cfg.fig2D2);set(gcf,'visible','off');
    screen_size=get(gcf,'Position');
    set(gcf,'Position',[screen_size(1) screen_size(2) fig_width fig_height]);
    cfg_est=strcat(plot_cfg.color,'-');
    subplot(plot_cfg.subgfig_row,plot_cfg.subgfig_col,plot_cfg.subgfig_idx);
    hold on;grid on;
    zero_vec = zeros(size(error_xy));
    draw_edges(zero_vec,error_xy,cfg_est);
    cfg_est=strcat(plot_cfg.color,'.');
    plot( error_xy(1,:), error_xy(2,:), cfg_est);
    error_xy_mean = mean(error_xy,2);
    tmp=[zeros(2,1) error_xy_mean];
    plot(tmp(1,:),tmp(2,:),'k-');
    plot(0,0,'ko');
    plot(error_xy_mean(1),error_xy_mean(2),'k.');
    title(plot_cfg.method_names(plot_cfg.subgfig_idx));
    xlabel('x [pixel]');
    ylabel('y [pixel]');
    box_half = 10;
    xlim([-box_half box_half]);
    ylim([-box_half box_half]);
   
    %% bias variance in polar coordinate
    % compute [r,theta] of the mean xy position;
    mean_r = norm(error_xy_mean);
    mean_theta = acos(error_xy_mean(1)/mean_r);
    if error_xy_mean(2)<0
        mean_theta = 2*pi -mean_theta;
    end
    mean_theta = mean_theta/pi*180;
    
    % plot errors in the polar coordinates
    figure(plot_cfg.fig2D3);set(gcf,'visible','off')
    screen_size=get(gcf,'Position');
    set(gcf,'Position',[screen_size(1) screen_size(2) fig_width fig_height]);
    subplot(plot_cfg.subgfig_row,plot_cfg.subgfig_col,plot_cfg.subgfig_idx);
    hold on;grid on;
    r      = error_r_theta(1,:);
    angles = error_r_theta(2,:)/pi*180;
    z = zeros(size(r));
    draw_edges([angles;z],[angles;r],strcat(plot_cfg.color,'-'));
    plot( angles, r, cfg_est);
    
    % plot mean value in polar coord.
    tmp=[ [mean_theta;0.0]  [mean_theta;mean_r] ];
    plot(tmp(1,:),tmp(2,:),'k-');
    %plot(mean_theta,0     ,'ko');
    plot(mean_theta,mean_r,'k.');
    
    title(plot_cfg.method_names(plot_cfg.subgfig_idx));
    xlabel('\theta [deg.]');
    ylabel('Length [pixel]');
    xlim([0 360]);
    ylim([0 20]);
        
   %% Correlations
    %plot_cfg.method_names(plot_cfg.subgfig_idx)
    %idx = plot_cfg.error_sets_label_idx;
    plot_cfg.pt_corr_GT_uv_vs_ErrorVector.set( [ plot_cfg.pt_corr_GT_uv_vs_ErrorVector.get() corr2(xy_gt',error_xy') ] );
    u=xy_gt(1,:);
    v=xy_gt(2,:);
    x=xyz_gt(1,:);
    y=xyz_gt(2,:);
    z=xyz_gt(3,:);
    plot_cfg.pt_corr_GT_u_vs_errror_r.set( [ plot_cfg.pt_corr_GT_u_vs_errror_r.get() corr2(u,r) ] );
    plot_cfg.pt_corr_GT_v_vs_errror_r.set( [ plot_cfg.pt_corr_GT_v_vs_errror_r.get() corr2(v,r) ] );
    plot_cfg.pt_corr_GT_x_vs_errror_r.set( [ plot_cfg.pt_corr_GT_x_vs_errror_r.get() corr2(x,r) ] );
    plot_cfg.pt_corr_GT_y_vs_errror_r.set( [ plot_cfg.pt_corr_GT_y_vs_errror_r.get() corr2(y,r) ] );
    plot_cfg.pt_corr_GT_z_vs_errror_r.set( [ plot_cfg.pt_corr_GT_z_vs_errror_r.get() corr2(z,r) ] );
    plot_cfg.pt_corr_GT_u_vs_errror_t.set( [ plot_cfg.pt_corr_GT_u_vs_errror_t.get() corr2(u,angles) ] );
    plot_cfg.pt_corr_GT_v_vs_errror_t.set( [ plot_cfg.pt_corr_GT_v_vs_errror_t.get() corr2(v,angles) ] );
    plot_cfg.pt_corr_GT_x_vs_errror_t.set( [ plot_cfg.pt_corr_GT_x_vs_errror_t.get() corr2(x,angles) ] );
    plot_cfg.pt_corr_GT_y_vs_errror_t.set( [ plot_cfg.pt_corr_GT_y_vs_errror_t.get() corr2(y,angles) ] );
    plot_cfg.pt_corr_GT_z_vs_errror_t.set( [ plot_cfg.pt_corr_GT_z_vs_errror_t.get() corr2(z,angles) ] );
    
   %% Eye positions boxplot
    s_num = plot_cfg.SequenceNum;
    c_num = 2;%plot_cfg.subgfig_col;
    % Only SPAAM and one of proposed method is visualized since eye positions
    % are only dependent on SPAAM or tracker
    if plot_cfg.subgfig_idx==1
        subplot_idx=1;
    elseif  plot_cfg.subgfig_idx==3
        subplot_idx=2;
    else
        return;
    end
    title_name = plot_cfg.method_names(plot_cfg.subgfig_idx);
    %%% For each sequence
    for s_idx = 1:s_num
        eye_xyz = eye_xyz_cell{s_idx};
        figure(plot_cfg.fig3D2);
        set(gcf,'visible','off')
        %screen_size=get(gcf,'Position');
        %set(gcf,'Position',[screen_size(1) screen_size(2) fig_width/2 fig_height]);
        subplot( s_num, c_num, c_num*(s_idx-1)+subplot_idx);
        set(gcf,'visible','off')
        %subplot(plot_cfg.subgfig_row,plot_cfg.subgfig_col,plot_cfg.subgfig_idx);
        hold on;grid on;
        eye_xyz_unique =unique(eye_xyz','rows');
        eye_xyz_unique = eye_xyz_unique';
        
        %plot3(eye_xyz(1,:),eye_xyz(2,:),eye_xyz(3,:),strcat(plot_cfg.color,'.'));
        num = size(eye_xyz_unique,2);
        eye_xyz_1Darray=eye_xyz_unique';
        eye_xyz_1Darray=eye_xyz_1Darray(:);
        labels = [ones(1,num), ones(1,num)*2, ones(1,num)*3];
        
        boxplot(eye_xyz_1Darray, labels, 'labels',{'x','y','z'});
        %title_name2 = strcat(title_name,', Seq. ',num2str(s_idx),' t_{EW}');
        title_name2 = strcat(title_name);
        title(title_name2);
        %title(plot_cfg.method_names(plot_cfg.subgfig_idx));
        ylim([-0.155 0.17]);
    end
end

%% Compute distorted 2D points
function uv_hd = distort(uv_hu,cx,cy,k1,k2,k3,p1,p2)
    c=[cx;cy]*ones(1,size(uv_hu,2));
    d_uv_hu = uv_hu - c;
    r_h2=[1;1]*sum(d_uv_hu.*d_uv_hu);
    r_h4=r_h2.*r_h2;
    r_h6=r_h4.*r_h2;
    du_dv = d_uv_hu(1,:).*d_uv_hu(2,:);
    p12=[p1;p2];
    tmp =  ( r_h2+2*(d_uv_hu.*d_uv_hu) );
    tmp(1,:) = p2*tmp(1,:); 
    tmp(2,:) = p1*tmp(2,:); 
    uv_hd =   d_uv_hu.*( 1 + k1*r_h2 + k2*r_h4 + k3*r_h6 )...
            + (2*p12)*du_dv + tmp + c;
end


