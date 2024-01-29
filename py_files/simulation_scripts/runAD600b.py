working_dir = '/mnt/c/users/fordd/Downloads/RW_PGM_FE/'
py_files_dir =  working_dir+'py_files/'
saved_arrays_dir = working_dir+'saved_arrays/'
pdb_files_dir = working_dir+'pdb_files/'
trajectory_files_dir = working_dir+'trajectory_files/'

##
import os
os.chdir(py_files_dir)
from utils import Single_Molecule_In_Vaccum
os.chdir(working_dir)
##

T = 600

PDB = pdb_files_dir+'alanine-dipeptide-in-vacuum.pdb'
inds_CVs = [[4, 6, 8, 14], [6, 8, 14, 16]]
FF_name = "amber99sbildn.xml"

smiv = Single_Molecule_In_Vaccum(PDB,
                                 FF=FF_name,
                                 default_temperature=T)

smiv.prepare_simulation_('AD'+str(T)+'b', 500000,
                         initial_conformer = smiv.pdb_conformer,
                         inds_torsional_CVs = inds_CVs,
                         FES_bandwidth = 0.07, 
                         FES_n_bins = 150,
                        )
smiv.run_simulation_(500000)

smiv.save_simulation_data_(trajectory_files_dir)

print('done AD'+str(T)+'b')
