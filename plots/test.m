%{
% First camera
R1 = eye(3);
t1 = [0 0 0];
rigid_pose1 = rigid3d(R1,t1);
plotCamera('AbsolutePose', rigid_pose1, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.05);
hold on;

% Second camera
T2 = [0.99991482, -0.012702983, 0.0029961311, 0.0013823361;
 0.012613448, 0.99952227, 0.02821625, -0.065855049;
 -0.0033531303, -0.028176054, 0.99959737, 0.012320181;
 0, 0, 0, 1];
T2 = inv(T2);
R2 = T2(1:3,1:3);
t2 = T2(1:3,4)';
rigid_pose2 = rigid3d(single(R2),t2);
plotCamera('AbsolutePose', rigid_pose2, 'Opacity', 0, ...
    "Color", "red", "Size", 0.1);

% Tirth camera
T3 = [0.99991757, -0.012492523, 0.0029621564, 0.0015686813;
 0.012410821, 0.99958074, 0.026158683, -0.065928973;
 -0.0032877026, -0.026119763, 0.9996534, 0.012956443;
 0, 0, 0, 1];
T3 = inv(T3);
R3 = T3(1:3,1:3);
t3 = T3(1:3,4)';
rigid_pose3 = rigid3d(single(R3),t2);
plotCamera('AbsolutePose', rigid_pose3, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.1);
xlabel("x"); ylabel("y"); zlabel("z");
view([0 -80]);
camproj('perspective');
waitforbuttonpress;
%}










%{
% First camera
t1 = [-0.9756 -0.8543 1.5687];
q1 = [-0.5739 0.5831 -0.4229 0.3895];
q1 = [q1(4) q1(1) q1(2) q1(3)];
R1 = quat2rotm(q1);
X1 = [R1, t1'; [0 0 0 1]]
pose1 = rigid3d(single(R1),t1);

plotCamera('AbsolutePose', pose1, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.05);
xlabel("x"); ylabel("y"); zlabel("z");

hold on;

% Second camera
t2 = [-0.9756 -0.8544 1.5694];
q2 = [-0.5738 0.5829 -0.4230 0.3899];
q2 = [q2(4) q2(1) q2(2) q2(3)];
R2 = quat2rotm(q2);
X2 = [R2, t2'; [0 0 0 1]]
pose2 = rigid3d(single(R2),t2);

plotCamera('AbsolutePose', pose2, 'Opacity', 0, ...
    "Color", "red", "Size", 0.1);
xlabel("x"); ylabel("y"); zlabel("z");
view([0 -80]);
camproj('perspective');
waitforbuttonpress;
%}









theta = 10;
%rot = [ cosd(theta) sind(theta) 0; ...
%       -sind(theta) cosd(theta) 0; ...
%       0 0 1];
rot = rotz(theta)
trans = [2 3 4];
tform = rigid3d(rot,trans);
plotCamera('AbsolutePose', tform, 'Opacity', 0, ...
                "Color", "red", "Size", 0.1);
xlabel("x"); ylabel("y"); zlabel("z");
view([0 -80]);
camproj('perspective');
waitforbuttonpress;