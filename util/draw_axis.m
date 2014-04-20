%% DRAW_AXIS draws a 3D coordinate system
%
% NOTE: This function turns on "hold" for the current figure pane
%
% DRAW_AXIS() simpliy draws a coordinate system at the origin
% DRAW_AXIS(R,t,s,color,kLineWidth)
% R: Orientation (3x3 unitray matrix)
% t: Position (3x1 vecoter)
% axes_length (lenght of each axis)
% color: 3x3 color matrix each row corresponds to an axis color 
% axes_width

% Copyright (c) Yuta Itoh 2014

function draw_axis(R,t,axes_length,color,axes_width)
hold on;

if nargin<=0
    R=eye(3);
    t=[0 0 0]';
    axes_length=0.5;
    color=eye(3);
axes_width=1;
elseif nargin<=1
    t=[0 0 0]';
    axes_length=0.5;
    color=eye(3);
axes_width=1;
elseif nargin<=2
    axes_length=0.5;
    color=eye(3);
axes_width=1;
elseif nargin<=3
    color=eye(3);
axes_width=1;
elseif nargin<=4
axes_width=1;
    
end
o=[0 0 0]'+t;
x=R*[axes_length 0 0]'+t;
y=R*[0 axes_length 0]'+t;
z=R*[0 0 axes_length]'+t;
tmp=[o x]';
cfg='-';
plot3(tmp(:,1),tmp(:,2),tmp(:,3),cfg,'LineWidth',axes_width,'color',color(1,:));
tmp=[o y]';
plot3(tmp(:,1),tmp(:,2),tmp(:,3),cfg,'LineWidth',axes_width,'color',color(2,:));
tmp=[o z]';
plot3(tmp(:,1),tmp(:,2),tmp(:,3),cfg,'LineWidth',axes_width,'color',color(3,:));
end