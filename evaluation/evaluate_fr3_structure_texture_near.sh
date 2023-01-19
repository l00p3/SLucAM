echo ATE ORB:
python2 evaluate_ate.py ../data/datasets/fr3_structure_texture_near/groundtruth.txt ../data/datasets/fr3_structure_texture_near/SLucAM_results_orb.txt --scale 0.95 --plot fr3_structure_texture_near/fr3_structure_texture_near_orb_ate_plot.png

echo RPE ORB:
python2 evaluate_rpe.py ../data/datasets/fr3_structure_texture_near/groundtruth.txt ../data/datasets/fr3_structure_texture_near/SLucAM_results_orb.txt --fixed_delta --scale 0.95 --verbose --plot fr3_structure_texture_near/fr3_structure_texture_near_orb_rpe_plot.png

echo

echo ATE Superpoint:
python2 evaluate_ate.py ../data/datasets/fr3_structure_texture_near/groundtruth.txt ../data/datasets/fr3_structure_texture_near/SLucAM_results_superpoint.txt --scale 0.7 --plot fr3_structure_texture_near/fr3_structure_texture_near_superpoint_ate_plot.png

echo RPE Superpoint:
python2 evaluate_rpe.py ../data/datasets/fr3_structure_texture_near/groundtruth.txt ../data/datasets/fr3_structure_texture_near/SLucAM_results_superpoint.txt --fixed_delta --scale 0.7 --verbose --plot fr3_structure_texture_near/fr3_structure_texture_near_superpoint_rpe_plot.png

echo

echo ATE LF-NET:
python2 evaluate_ate.py ../data/datasets/fr3_structure_texture_near/groundtruth.txt ../data/datasets/fr3_structure_texture_near/SLucAM_results_lf_net.txt --scale 0.8 --plot fr3_structure_texture_near/fr3_structure_texture_near_lf_net_ate_plot.png

echo RPE LF-NET:
python2 evaluate_rpe.py ../data/datasets/fr3_structure_texture_near/groundtruth.txt ../data/datasets/fr3_structure_texture_near/SLucAM_results_lf_net.txt --fixed_delta --scale 0.8 --verbose --plot fr3_structure_texture_near/fr3_structure_texture_near_lf_net_rpe_plot.png
