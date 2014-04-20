function A = openUbitrack3x4MatrixCalib( filename )
%OPENUBITRACK3X4MATRIXCALIB この関数の概要をここに記述
%   詳細説明をここに記述

fid = fopen( filename );
tline = fgetl( fid );
values = sscanf( tline, '%*d%*s%*d%*d%*d%*d%*d%*d%f%f%f%f%f%f%f%f%f%f%f%f' );
A = reshape( values , 4, 3 )';

fclose( fid );
end
