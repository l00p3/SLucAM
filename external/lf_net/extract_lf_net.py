import numpy as np
import argparse



# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("dataset_folder", help="Relative or absolute path to the dataset (with a / at the end of the name)")
parser.add_argument("lf_net_outputs_folder", help="Folder where the lf_net extracts the informations about images (with a / at the end of the name)")
args = parser.parse_args()

data_names_filename = args.dataset_folder + "rgb.txt"
extracted_data_folder = args.lf_net_outputs_folder
output_folder = args.dataset_folder+ "lf_net/"


# Load the list of images
data_names = []
f = open(data_names_filename, "r")
f.readline()    # Ignore the first three lines
f.readline()
f.readline()
for line in f:
    data_names.append(line.split()[0])
f.close()

# For each image
i = 1
n_imgs = len(data_names)
for data_name in data_names:

    # Load the info extracted for the current image
    current_infos = np.load(extracted_data_folder+data_name+".png.npz")

    # Take keypoints and descriptors
    kpts = current_infos['kpts']
    descs = current_infos['descs']

    # Save them on a file
    f = open(output_folder+data_name+".dat", "w")
    for kpt, desc in zip(kpts, descs):
        f.write(str(kpt[0]) + " " + str(kpt[1]) + " ")
        for el in desc:
            f.write(str(el)+" ")
        f.write("\n")
    f.close()

    print("Extracted image " + str(i) + "/" + str(n_imgs))
    i+=1

