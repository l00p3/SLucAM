function points = load_image_points(filename)

    % Open the file
    f = fopen(filename);
    
    % Ignore the header
    line = fgetl(f);
    line = fgetl(f);
    
    % Load the landmarks
    points = [];
    i = 1;
    line = fgetl(f);
    while ischar(line)
    
        line = strsplit(line);
        points(:,i) = [str2double(line{1});
                        str2double(line{2})];
        line = fgetl(f);
        i = i+1;
    
    end
    
    % Close the file
    fclose(f);

end