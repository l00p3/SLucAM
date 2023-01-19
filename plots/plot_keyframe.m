function res = plot_keyframe(folder, idx)

    % Initialization
    base_filename = folder + "frame";
    image_name_filename = "_SLucAM_image_name.dat";
    landmarks_filename = "_SLucAM_landmarks.dat";
    poses_filename = "_SLucAM_poses.dat";
    edges_filename = "_SLucAM_edges.dat";
    image_points_filename = "_SLucAM_img_points.dat";

    % Load poses, image and image's points
    poses = load_poses(base_filename+int2str(idx)+poses_filename);
    landmarks = load_landmarks(base_filename+int2str(idx)+landmarks_filename);
    img_filename = load_img_filename(base_filename+int2str(idx)+image_name_filename);
    edges = load_edges(base_filename+int2str(idx)+edges_filename);
    img_points = load_image_points(base_filename+int2str(idx)+image_points_filename);

    % Set some prior parameter for plot
    set(gcf, 'Position', get(0, 'Screensize'));
    t = tiledlayout(3,4);

    % Plot the image
    nexttile(1, [1 1]);
    img = imread(img_filename);
    imshow(img);
    hold on;

    % Plot image points
    scatter(img_points(1,:),img_points(2,:));
    title("Current measurement");
    hold on;
    
    % Plot pose graphs
    nexttile(5, [1,1]);
    plot_pose_graph(poses, "top");
    xlabel("x");
    ylabel("y");
    set(gca, 'Ydir', 'reverse')
    title("Pose graph (side view)");

    nexttile(9, [1,1]);
    plot_pose_graph(poses, "side");
    xlabel("x");
    ylabel("z");
    title("Pose graph (top view)");

    % Plot poses and landmarks
    nexttile(2, [3 3]);
    plot_poses(poses);
    hold on;
    scatter3(landmarks(1,:), landmarks(2,:), landmarks(3,:), 10, [0 0.6 1]);   % plot landmarks
    hold on;
    plot_edges(poses, edges, landmarks);
    hold on;
    %set(gca,'XColor','none','YColor','none','ZColor','none');
    xlabel("x"); ylabel("y"); zlabel("z");
    %view([0 80]);
    view([0 -80]);
    %axis([-10 10 -10 10 -0.3 30])
    axis([-4.5 2 -3 3 -3 5.5])
    camproj('perspective');
    title("World");
    
    % Set some posterior parameters for plot
    t.TileSpacing = 'compact';
    t.Padding = 'compact';

end