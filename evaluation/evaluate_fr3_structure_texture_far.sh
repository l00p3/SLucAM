echo ATE ORB:
python2 evaluate_ate.py ../data/datasets/fr3_structure_texture_far/groundtruth.txt ../data/datasets/fr3_structure_texture_far/SLucAM_results_orb.txt --scale 1 --plot fr3_structure_texture_far/fr3_structure_texture_far_orb_ate_plot.png

echo RPE ORB:
python2 evaluate_rpe.py ../data/datasets/fr3_structure_texture_far/groundtruth.txt ../data/datasets/fr3_structure_texture_far/SLucAM_results_orb.txt --fixed_delta --scale 1 --verbose --plot fr3_structure_texture_far/fr3_structure_texture_far_orb_rpe_plot.png

echo

echo ATE Superpoint:
python2 evaluate_ate.py ../data/datasets/fr3_structure_texture_far/groundtruth.txt ../data/datasets/fr3_structure_texture_far/SLucAM_results_superpoint.txt --scale 0.95 --plot fr3_structure_texture_far/fr3_structure_texture_far_superpoint_ate_plot.png

echo RPE Superpoint:
python2 evaluate_rpe.py ../data/datasets/fr3_structure_texture_far/groundtruth.txt ../data/datasets/fr3_structure_texture_far/SLucAM_results_superpoint.txt --fixed_delta --scale 0.95 --verbose --plot fr3_structure_texture_far/fr3_structure_texture_far_superpoint_rpe_plot.png

echo

echo ATE LF-NET:
python2 evaluate_ate.py ../data/datasets/fr3_structure_texture_far/groundtruth.txt ../data/datasets/fr3_structure_texture_far/SLucAM_results_lf_net.txt --scale 0.9 --plot fr3_structure_texture_far/fr3_structure_texture_far_lf_net_ate_plot.png

echo RPE LF-NET:
python2 evaluate_rpe.py ../data/datasets/fr3_structure_texture_far/groundtruth.txt ../data/datasets/fr3_structure_texture_far/SLucAM_results_lf_net.txt --fixed_delta --scale 0.9 --verbose --plot fr3_structure_texture_far/fr3_structure_texture_far_lf_net_rpe_plot.png
