%%  Decompose a 3x4 projection matrix into K, R, and t
%   
%   K : 3x3 intrinsic matrix
%   R : 3x3 rotation matrix
%   t : 3x1 translation vector
%  
%   Note that this function assumes the RIGHT-handed coordinate system:
%   K = [fx  0 -cx; 
%         0 fy -cy;
%         0  0 -1]
%   where (fx,fy) and (cx,cy) are 
%   focal lengths and pixel center respectively.

% Copyright (c) Yuta Itoh 2014

function [K,R,t] = decompose_projection(P)

[K, R] = rq(P(:,1:3));
%S=eye(3);
S = diag([sign(K(1,1)),sign(K(2,2)),-sign(K(3,3))]);
R = S\R;%inv(S)*Rgt;
K = K*S;
t = K\P(:,4);%inv(Kgt)*P(:,4);
end