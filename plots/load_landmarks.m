function landmarks = load_landmarks(filename)

    % Open the file
    f = fopen(filename);
    
    % Ignore the header
    line = fgetl(f);
    line = fgetl(f);
    
    % Load the landmarks
    landmarks = [];
    i = 1;
    line = fgetl(f);
    while ischar(line)
    
        line = strsplit(line);
        landmarks(:,i) = [str2double(line{1});
                        str2double(line{2});
                        str2double(line{3})];
        line = fgetl(f);
        i = i+1;
    
    end
    
    % Close the file
    fclose(f);

end