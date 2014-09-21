%% INDICA_Recycle Interaction-free Calibration (Full Setup)
%
% Implementation of eye position-based calibration method (Full Setup)
% for Interaction-free calibtation of

% Copyright (c) Yuta Itoh 2014

function [P_WE,t_WE] = INDICA_Full(R_WS,R_WT,t_WT,t_ET,t_WS,ax,ay,w,h)
t_WE = -R_WS*(-R_WT'*t_WT+ R_WT'*t_ET);

% adjust pixel center
Q=[ 1 0 w/2-0.5; 
    0 1 h/2-0.5;
    0 0 1];
b  = (t_WE - t_WS);
bx = b(1);
by = b(2);
bz = b(3);
K = [ bz 0 -bx;
      0 bz -by; 
      0  0  1];
P_WE = Q*diag([ax ay 1])*K*[R_WS t_WE ];
end
