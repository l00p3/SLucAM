function res = plot_poses(poses)

    for i = 1:size(poses,3)
        
        pose = poses(:,:,i);
        R = [pose(1,1), pose(1,2), pose(1,3); ...
            pose(2,1), pose(2,2), pose(2,3); ...
            pose(3,1), pose(3,2), pose(3,3)];
        t = [pose(1,4) pose(2,4) pose(3,4)];
        rigid_pose = rigid3d(single(R'),t);
        if i == size(poses,3)
            cam = plotCamera('AbsolutePose', rigid_pose, 'Opacity', 0, ...
                "Color", "red", "Size", 0.1);
        else
            cam = plotCamera('AbsolutePose', rigid_pose, 'Opacity', 0, ...
                "Color", "blue", "Size", 0.03);
        end
        hold on;

    end

    res = 1;

end