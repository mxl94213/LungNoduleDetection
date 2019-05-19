from glob import glob

workpath = '../data/output_files/'
ifile_list = glob(workpath+"images_*.npy")
print(ifile_list)