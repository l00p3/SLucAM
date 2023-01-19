echo ATE ORB:
python2 evaluate_ate.py ../data/datasets/fr2_desk/groundtruth.txt ../data/datasets/fr2_desk/SLucAM_results_orb.txt --scale 1.35 --plot fr2_desk/fr2_desk_orb_ate_plot.png

echo RPE ORB:
python2 evaluate_rpe.py ../data/datasets/fr2_desk/groundtruth.txt ../data/datasets/fr2_desk/SLucAM_results_orb.txt --fixed_delta --scale 1.35 --verbose --plot fr2_desk/fr2_desk_orb_rpe_plot.png

echo

echo ATE Superpoint:
python2 evaluate_ate.py ../data/datasets/fr2_desk/groundtruth.txt ../data/datasets/fr2_desk/SLucAM_results_superpoint.txt --scale 0.95 --plot fr2_desk/fr2_desk_superpoint_ate_plot.png

echo RPE Superpoint:
python2 evaluate_rpe.py ../data/datasets/fr2_desk/groundtruth.txt ../data/datasets/fr2_desk/SLucAM_results_superpoint.txt --fixed_delta --scale 0.6 --verbose --plot fr2_desk/fr2_desk_superpoint_rpe_plot.png

echo

echo ATE LF-NET:
python2 evaluate_ate.py ../data/datasets/fr2_desk/groundtruth.txt ../data/datasets/fr2_desk/SLucAM_results_lf_net.txt --scale 0.7 --plot fr2_desk/fr2_desk_lf_net_ate_plot.png

echo RPE LF-NET:
python2 evaluate_rpe.py ../data/datasets/fr2_desk/groundtruth.txt ../data/datasets/fr2_desk/SLucAM_results_lf_net.txt --fixed_delta --scale 0.7 --verbose --plot fr2_desk/fr2_desk_lf_net_rpe_plot.png
