echo ATE ORB:
python2 evaluate_ate.py ../data/datasets/fr2_xyz/groundtruth.txt ../data/datasets/fr2_xyz/SLucAM_results_orb.txt --scale 0.65 --plot fr2_xyz/fr2_xyz_orb_ate_plot.png

echo RPE ORB:
python2 evaluate_rpe.py ../data/datasets/fr2_xyz/groundtruth.txt ../data/datasets/fr2_xyz/SLucAM_results_orb.txt --fixed_delta --scale 0.65 --verbose --plot fr2_xyz/fr2_xyz_orb_rpe_plot.png

echo

echo ATE Superpoint:
python2 evaluate_ate.py ../data/datasets/fr2_xyz/groundtruth.txt ../data/datasets/fr2_xyz/SLucAM_results_superpoint.txt --scale 0.7 --plot fr2_xyz/fr2_xyz_superpoint_ate_plot.png

echo RPE Superpoint:
python2 evaluate_rpe.py ../data/datasets/fr2_xyz/groundtruth.txt ../data/datasets/fr2_xyz/SLucAM_results_superpoint.txt --fixed_delta --scale 0.7 --verbose --plot fr2_xyz/fr2_xyz_superpoint_rpe_plot.png

echo

echo ATE LF-NET:
python2 evaluate_ate.py ../data/datasets/fr2_xyz/groundtruth.txt ../data/datasets/fr2_xyz/SLucAM_results_lf_net.txt --scale 0.75 --plot fr2_xyz/fr2_xyz_lf_net_ate_plot.png

echo RPE LF-NET:
python2 evaluate_rpe.py ../data/datasets/fr2_xyz/groundtruth.txt ../data/datasets/fr2_xyz/SLucAM_results_lf_net.txt --fixed_delta --scale 0.75 --verbose --plot fr2_xyz/fr2_xyz_lf_net_rpe_plot.png
