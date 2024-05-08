function [elem, node, face] = read_tetra(filename)
%function [elem, node] = read_tetra(filename)
%%% node
fid = fopen([filename,'.node'],'r');
if( fid==-1 )
    error('Cannot open the file.');
    return;
end
str = fgets(fid);
[node_size,dim] = strtok(str); 
[A,cnt] = fscanf(fid,'%d %f %f %f');

A = reshape(A, 4,[]);
node = A(2:4,:);
fclose(fid);

%%% elem
fid = fopen([filename,'.ele'],'r');
if( fid==-1 )
    error('Can''t open the file.');
    return;
end
str = fgets(fid);
ele_size = strtok(str); 
[A,cnt] = fscanf(fid,'%d %d %d %d %d %d');

A = reshape(A, 6,[]);
elem = A(2:5,:);
elem(1:4,:)= elem(1:4,:)+1; 
fclose(fid);
%%%%% face
fid = fopen([filename,'.face'],'r');
if( fid==-1 )
    error('Can''t open the file.');
    return;
end
str = fgets(fid);
[face_size,class] = strtok(str); 
[A,cnt] = fscanf(fid,'%d %d %d %d %d\n');

A = reshape(A, 5,[]);
face = A(2:4,:);
%face = A(2:5,:)+1;
face(1:3,:) = face(1:3,:) + 1;
fclose(fid);

node = node';
elem = elem';
face = face';




