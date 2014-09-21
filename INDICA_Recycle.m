%% INDICA_Recycle Interaction-free Calibration (Recycle Setup)
%
% Implementation of eye position-based calibration method (Recycle Setup)
% for Interaction-free calibtation of

% Copyright (c) Yuta Itoh 2014

function [P_WE,t_WE] = INDICA_Recycle(R_WS,R_WT,t_WT,t_ET,t_WS_z,K_E0,t_WE0)
t_WE = -R_WS*(-R_WT'*t_WT+ R_WT'*t_ET);
%t_WE = t_WE  + R_WS'*[0, -0.1323-(-0.1400), 0.0949 - 0.1008]';


t_SE0_z =(t_WE0(3)- t_WS_z );
t_E0E = t_WE-t_WE0;% eye2 position
K2=[ (t_SE0_z+t_E0E(3))  0    -t_E0E(1)
    0 (t_SE0_z+t_E0E(3)) -t_E0E(2)
    0         0          t_SE0_z ];
K2=K2/t_SE0_z;
P_WE = K_E0*K2*[R_WS t_WE];
end
