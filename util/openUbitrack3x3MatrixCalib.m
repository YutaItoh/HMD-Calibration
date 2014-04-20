function A = openUbitrack3x3MatrixCalib( filename )

fid = fopen( filename );
tline = fgetl( fid );
values = sscanf( tline, '%*d%*s%*d%*d%*d%*d%*d%*d%f%f%f%f%f%f%f%f%f' );
A = reshape( values , 3, 3 )';
fclose( fid );


% 22 serialization::archive 7 0 0 1296581878919480452 0 0 451.944517247992 0 -341.393089216867 0 450.247365208464 -259.327605219859 0 0 -1
