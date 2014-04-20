%% DLT Direct Linear Transform
% 
% Compute 3x4 projection matrix P by Direct Linear Transform
% so that uv ~= P*[xyz;ones] 
%
% P = DLT(uv,xyz)
% uv:  2xN matrix
% xyz: 3xN matrix
% P: 3x4 projection matrix
%
% Make sure N>=6
% 

% Copyright (c) Yuta Itoh 2014

function P = DLT(uv, xyz)
N=size(uv,2);
if N~=size(xyz,2)
    error('DLT: Number of samples must be same for xy and xyz');
end

% Estimate projection parameter
B=zeros(2*N,12);
xyz=[xyz;ones(1,N)];
for i=1:N
    B(2*i-1,:)=[xyz(:,i)' 0 0 0 0 -uv(1,i)*xyz(:,i)'];
    B(2*i  ,:)=[0 0 0 0 xyz(:,i)' -uv(2,i)*xyz(:,i)'];
end
[~,~,V] = svd(B);
g=V(:,12);
Gest=[ g(1:4)'; g(5:8)'; g(9:12)'];
Gest=Gest/Gest(3,4);
P=Gest;
end

