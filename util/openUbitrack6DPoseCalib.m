function A = openUbitrack6DPoseCalib( filename )

fid = fopen( filename );
tline = fgetl( fid );
A = sscanf( tline, '%*d%*s%*d%*d%*d%f%*d%*d%*d%*d%f%f%f%f%*d%*d%f%f%f');
fclose( fid );

%22 serialization::archive 7 0 0 1288184417219382801 0 0 0 0 0.67721901589513189 0.22927133973975339 0.2306315983876332 0.6600137294853291 0 0 -1.2301900665597256 0.11401148036265873 -0.094900924811221615
