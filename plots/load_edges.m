function edges = load_edges(filename)

    % Open the file
    f = fopen(filename);
    
    % Ignore the header
    line = fgetl(f);
    line = fgetl(f);
    
    % Load the edges
    edges = [];
    line = fgetl(f);
    while ischar(line)
    
        edges(end+1) = str2double(line);
        line = fgetl(f);
    
    end
    
    % Close the file
    fclose(f);

end