function res = plot_pose_graph(poses, view)

    positions = [];

    for i = 2:size(poses,3)
    
        pose = poses(:,:,i);
        positions(:,end+1) = [pose(1,4); pose(2,4); pose(3,4)];

    end
    
    if(view == "top")
        plot(positions(1,1:end-1), positions(2,1:end-1), ':x');
        hold on;
        plot(positions(1,end), positions(2,end), 'x', "Color","red");
    end

    if(view == "side")
        plot(positions(1,1:end-1), positions(3,1:end-1), ':x');
        hold on;
        plot(positions(1,end), positions(3,end), 'x', "Color","red");
    end

    res = 1;

end