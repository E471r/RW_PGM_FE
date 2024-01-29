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

T = 300

PDB = pdb_files_dir+'ibuprofen-in-vacuum.pdb'
inds_CVs = [[13, 10, 3, 1], [4, 3, 1, 0]]
FF_name = pdb_files_dir+"ibuprofen-in-vacuum-system.prmtop"

smiv = Single_Molecule_In_Vaccum(PDB,
                                 FF=FF_name,
                                 default_temperature=T)

smiv.prepare_simulation_('IB'+str(T)+'b', 1000000,
                         initial_conformer = smiv.pdb_conformer,
                         inds_torsional_CVs = inds_CVs,
                         FES_bandwidth = 0.06,
                         FES_n_bins = 175, 
                        )
smiv.run_simulation_(1000000)

smiv.save_simulation_data_(trajectory_files_dir)

print('done IB'+str(T)+'b')
