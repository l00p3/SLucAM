function res = plot_edges(poses, edges, landmarks)

    % Initialization
    n_edges = size(edges,2);
    last_pose = poses(:,:,end);
    t = [last_pose(1,4); ...
          last_pose(2,4); ...
          last_pose(3,4)];
    
    % Plot all edges
    for i = 1:n_edges

        % Take the current landmark
        l = landmarks(:,edges(i)+1);

        % Plot the line
        plot3([t(1), l(1)], ...
              [t(2), l(2)], ...
              [t(3), l(3)], ':');
        hold on;

    end

end