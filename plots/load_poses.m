function poses = load_poses(filename)

    % Open the file
    f = fopen(filename);
    
    % Ignore the header
    line = fgetl(f);
    line = fgetl(f);
    
    % Load the poses
    poses = [];
    i = 1;
    line = fgetl(f);
    while ischar(line)
        
        line = strsplit(line);

        pose = eye(4);
        pose(1,4) = str2double(line{1});
        pose(2,4) = str2double(line{2});
        pose(3,4) = str2double(line{3});
        pose(1,1) = str2double(line{4});
        pose(1,2) = str2double(line{5});
        pose(1,3) = str2double(line{6});
        pose(2,1) = str2double(line{7});
        pose(2,2) = str2double(line{8});
        pose(2,3) = str2double(line{9});
        pose(3,1) = str2double(line{10});
        pose(3,2) = str2double(line{11});
        pose(3,3) = str2double(line{12});
        
        % inv(pose) because we saved them as world wrt cam
        % and here we need cam wrt world
        poses(:,:,i) = inv(pose);
        
        line = fgetl(f);
        i = i+1;
    
    end
    
    % Close the file
    fclose(f);

end