function A = openUbitrack3x4MatrixCalib( filename )
%OPENUBITRACK3X4MATRIXCALIB ���̊֐��̊T�v�������ɋL�q
%   �ڍא����������ɋL�q

fid = fopen( filename );
tline = fgetl( fid );
values = sscanf( tline, '%*d%*s%*d%*d%*d%*d%*d%*d%f%f%f%f%f%f%f%f%f%f%f%f' );
A = reshape( values , 4, 3 )';

fclose( fid );
end
