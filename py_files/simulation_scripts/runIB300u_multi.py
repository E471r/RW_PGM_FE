working_dir = '/mnt/c/users/fordd/Downloads/RW_PGM_FE/'
py_files_dir =  working_dir+'py_files/'
saved_arrays_dir = working_dir+'saved_arrays/'
pdb_files_dir = working_dir+'pdb_files/'
trajectory_files_dir = working_dir+'trajectory_files/'

##
import os
os.chdir(py_files_dir)
from utils import np, Single_Molecule_In_Vaccum, load_pickle_, save_pickle_
os.chdir(working_dir)
##

n_save = 25000

T = 300

PDB = pdb_files_dir+'ibuprofen-in-vacuum.pdb'
FF_name = pdb_files_dir+"ibuprofen-in-vacuum-system.prmtop"

n_states = 6

r_init = load_pickle_(saved_arrays_dir+'IB300u_init_ki_xyz')

xyz_ki = []
u_ki = []
Tu_ki = []
for k in range(n_states):
    for i in range(10):
        smiv = Single_Molecule_In_Vaccum(PDB,
                                         FF=FF_name,
                                         default_temperature=T)
        smiv.prepare_simulation_('IB_'+str(T)+'K', n_save+50,
                                 initial_conformer = r_init[k,i],
                                )
        print('ready to run k=',k+1,'/6, i=',i+1,'/10')
        print('n_save=',n_save)
        print('_____________________________________________')
        print('_____________________________________________')
        smiv.run_simulation_(n_save+50)
        print('storing')
        xyz_ki.append(smiv.xyz[50:])
        u_ki.append(smiv.u[50:])
        Tu_ki.append(smiv.Tu[50:])

save_pickle_(np.concatenate(xyz_ki,axis=0), trajectory_files_dir+'IB300u_sim_ki_xyz')
save_pickle_(np.concatenate(u_ki,axis=0), trajectory_files_dir+'IB300u_sim_ki_u')
save_pickle_(np.concatenate(Tu_ki,axis=0), trajectory_files_dir+'IB300u_sim_ki_Tu')

print('done IB'+str(T)+'u_multi')