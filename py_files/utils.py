import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle

from IC_MAP import get_torsion_tf_

##

def save_pickle_(x,name):
    with open(name, "wb") as f: pickle.dump(x, f) ; print('saved',name)
    
def load_pickle_(name):
    with open(name, "rb") as f: x = pickle.load(f) ; return x

##

PI = np.pi

DTYPE_tf = tf.float32
tf2np_ = lambda x : x.numpy() 
np2tf_ = lambda x : tf.cast(x, dtype=DTYPE_tf)

##

def get_torsional_CVs_(xyz, inds_CVs):
    n_CVs = len(inds_CVs)
    return np.concatenate([tf2np_(get_torsion_tf_(np2tf_(xyz), inds_CVs[i])[0]) for i in range(n_CVs)], axis=-1)

##

def import_openmm():
    try:
        import openmm as mm
        import openmm.unit as unit
        import openmm.app as app
    except ImportError:
        from simtk import openmm as mm
        from simtk import unit as unit
        from simtk.openmm import app as app
    return mm, unit, app

mm, unit, app = import_openmm()
from multicontext_openmm import MultiContext # Reference: https://github.com/noegroup/bgflow
from WTmetaD_edited import wtmetad, BiasVariable
import os

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

class Single_Molecule_In_Vaccum:
    def __init__(self,
                 PDB : str, # file name of .pdb file of molecule to initialise the topology.
                 FF : str = "amber99sbildn.xml", # or .prmtop file.
                 custom_n_cores : int = None,
                 default_temperature : int = 300, # K
                 ):
        self.PDB_path = PDB
        self.FF_name = FF
        self.custom_n_cores = custom_n_cores
        self.default_temperature = default_temperature
        self.kB = 1e-3*8.31446261815324 # kilojoule/(kelvin*mole)
        ##

        self.pdb = app.PDBFile(self.PDB_path) ; self.n_atoms = len(self.pdb.positions) ; self.c = (2./(3.*self.n_atoms*self.kB))
        self.pdb_conformer = self.pdb.getPositions(asNumpy=True)._value
        
        try:
            self.ff = app.ForceField(self.FF_name)
            print('using app.ForceField')
            self.reset_system_ = self.reset_system_I_
        except:
            self.ff = app.AmberPrmtopFile(self.FF_name)
            print('using app.AmberPrmtopFile')
            self.reset_system_ = self.reset_system_II_

        self.reset_system_()
        
        if custom_n_cores is not None: self.n_cores_eval = custom_n_cores
        else:
            self.n_cpu_cores = os.cpu_count()
            print('# cpu cores will be used for evaluation:', self.n_cpu_cores)
            self.n_cores_eval = self.n_cpu_cores

        # here intended only for batch evaluation, not for simulations.
        self.MC = MultiContext(n_workers = self.n_cores_eval,
                               system = self.system,
                               integrator = mm.LangevinIntegrator(0,0,0),
                               platform_name = 'CPU')
        
        self.simulation_initialised = False
        
    def reset_system_I_(self,):
        self.system = self.ff.createSystem(topology=self.pdb.getTopology(),
                                           removeCMMotion=True,
                                           nonbondedMethod=app.NoCutoff,
                                           constraints=None)
    def reset_system_II_(self,):
        self.system = self.ff.createSystem(removeCMMotion=True,
                                           nonbondedMethod=app.NoCutoff,
                                           constraints=None)
        
    def u_(self,
           x, # (m,n,3)
           T = None
           ):
       ' reduced potential energy function for evaluation purposes only '
       if T is None: T = self.default_temperature
       else: pass
       beta = 1.0 / (self.kB*T)
       u = beta * self.MC.evaluate(np.array(x), evaluate_force=False)[0][:,np.newaxis]
       return u # (m,1)

    def prepare_simulation_(self,
                            name : str,
                            max_frames_wanted,
                            T = None,
                            temestep_ps = 0.002,
                            initial_conformer = None, # (n,3) array
                            initial_conformer_units = unit.nanometers,
                            inds_torsional_CVs : list = None, # triggers biased mode. has to be 2 torsion for now!
                            biased_stride = 500,
                            unbased_stride = 100,
                            FES_bandwidth = 0.07, # 0.03
                            FES_n_bins = 150,     # 300 
                            ):
        self.simulation_name = name
        ' before either of the *two types of simulations below '
        if T is None:  self.T_of_simulation = self.default_temperature
        else: self.T_of_simulation = T
        self.temestep_ps = temestep_ps
        self.reset_system_()
        self.integrator = mm.LangevinIntegrator(self.T_of_simulation*unit.kelvin,
                                                1.0/unit.picosecond,
                                                self.temestep_ps*unit.picoseconds)
        self.simulation = app.Simulation(self.pdb.topology, self.system, self.integrator)

        if initial_conformer is None: initial_conformer = self.pdb.positions
        else: initial_conformer = initial_conformer * initial_conformer_units

        self.simulation.context.setPositions(initial_conformer)
        self.initial_conformer = self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        self.simulation.minimizeEnergy()
        self.minimised_initial_conformer = self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        
        self.sim_xyz = np.zeros([max_frames_wanted,self.n_atoms,3])
        self.sim_s = np.zeros([max_frames_wanted,2])
        self.sim_u = np.zeros([max_frames_wanted,1])
        self.sim_v = np.zeros([max_frames_wanted,1])
        self.sim_Tu = np.zeros([max_frames_wanted,1])
        self.sim_Tv = np.zeros([max_frames_wanted,1])
        self.max_frames_wanted = max_frames_wanted

        self.KtoT_ = lambda K : self.c * K
        self.frames_saved = 0

        #

        self.stride_biased = biased_stride
        self.stride_unbiased = unbased_stride

        ' the *two types of simulations are: biased or unbiased '
        ' biasing is via WTmetaD method on 2 periodic CVs (via WTmetaD_edited.py) '
        if inds_torsional_CVs is not None:
            self.biased = True
            self.inds_torsional_CVs = np.array(inds_torsional_CVs).tolist()
            print('CVs are torsions:',self.inds_torsional_CVs[0],'and', self.inds_torsional_CVs[1])

            cv1 = mm.CustomTorsionForce('theta')
            cv1.addTorsion(*self.inds_torsional_CVs[0])
            
            cv2 = mm.CustomTorsionForce('theta')
            cv2.addTorsion(*self.inds_torsional_CVs[1])

            CV1 = BiasVariable(cv1, -np.pi, np.pi, FES_bandwidth, True, FES_n_bins)
            CV2 = BiasVariable(cv2, -np.pi, np.pi, FES_bandwidth, True, FES_n_bins)
            self.WTsim = wtmetad(self.system, [CV1, CV2],
                                 self.T_of_simulation*unit.kelvin,
                                 3000,#*unit.kelvin,              # deltaT 
                                 0.5, #*unit.kilojoules_per_mole, # h0
                                 frequency = self.stride_biased)
            ## 
            self.integrator = mm.LangevinIntegrator(self.T_of_simulation*unit.kelvin,
                                        1.0/unit.picosecond,
                                        self.temestep_ps*unit.picoseconds)
            self.simulation = app.Simulation(self.pdb.topology, self.system, self.integrator)
            self.simulation.context.setPositions(self.minimised_initial_conformer)
            ##

            self.run_simulation_ = self.simulate_WTmetaD_
            print('ready to run_simulation_() (biased) at T=',self.T_of_simulation,'K, of',self.n_atoms,'atoms.')
            self.simulation_initialised = True
            print('')
        else:
            self.biased = False
            self.run_simulation_ = self.simulate_unbiased_
            print('ready to run_simulation_() (unbiased) at T=',self.T_of_simulation,'K, of',self.n_atoms,'atoms.')
            self.simulation_initialised = True
            print('')

    def simulate_unbiased_(self,
                           n_saves, # e.g., 1M*unbased_stride
                          ):
        kT = self.kB*self.T_of_simulation
        for i in range(n_saves):
            ## TODO: make this block ummune to interuptions.
            self.simulation.step(self.stride_unbiased)
            self.sim_xyz[self.frames_saved] = self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
            self.sim_u[self.frames_saved] = self.simulation.context.getState(getEnergy=True, groups={0}).getPotentialEnergy()._value / kT
            self.sim_Tu[self.frames_saved] = self.KtoT_(self.simulation.context.getState(getEnergy=True, groups={0}).getKineticEnergy()._value)
            # ^units: nm, kT, T

            #print ("\033[A                             \033[A")
            clear_output(wait=True)
            print(100*self.frames_saved/self.max_frames_wanted,'%','before need to save_simulation_data_()')

            self.frames_saved += 1

    def simulate_WTmetaD_(self,
                          n_saves, # e.g., 1M*unbased_stride
                          ):
        for i in range(n_saves):
            ## TODO: make this block ummune to interuptions.
            self.WTsim.step(self.simulation, self.stride_biased)
            self.sim_xyz[self.frames_saved] = self.WTsim.list_coordinates[0] # nm
            self.sim_s[self.frames_saved] = self.WTsim.list_CV[0] # radians
            self.sim_u[self.frames_saved] = self.WTsim.list_u[0] # kT
            self.sim_v[self.frames_saved] = self.WTsim.list_v[0] # kT
            self.sim_Tu[self.frames_saved] = self.KtoT_(self.WTsim.list_K_u[0]) # K
            self.sim_Tv[self.frames_saved] = self.KtoT_(self.WTsim.list_K_v[0]) # K

            #print ("\033[A                             \033[A")
            clear_output(wait=True)
            print(100*self.frames_saved/self.max_frames_wanted,'%','before need to save_simulation_data_()')

            self.frames_saved += 1

    @property
    def simulation_timescale(self,):
        if self.biased: _time = self.stride_biased*self.frames_saved*self.temestep_ps/1000.
        else:           _time = self.stride_unbiased*self.frames_saved*self.temestep_ps/1000.
        print(_time, 'ns')
        return _time

    @property
    def xyz(self,):
        if self.simulation_initialised: return np.array(self.sim_xyz[:self.frames_saved])
        else: return np.array(self._xyz)
    @property
    def s(self,):
        if self.simulation_initialised: return np.array(self.sim_s[:self.frames_saved])
        else: return np.array(self._s)
    @property
    def u(self,):
        if self.simulation_initialised: return np.array(self.sim_u[:self.frames_saved])
        else: return np.array(self._u)
    @property
    def v(self,):
        if self.simulation_initialised: return np.array(self.sim_v[:self.frames_saved])
        else: return np.array(self._v)
    @property
    def Tu(self,):
        if self.simulation_initialised: return np.array(self.sim_Tu[:self.frames_saved])
        else: return np.array(self._Tu)
    @property
    def Tv(self,):
        if self.simulation_initialised: return np.array(self.sim_Tv[:self.frames_saved])
        else: return np.array(self._Tv)
    @property
    def FES(self,):
        if self.simulation_initialised:
            if self.biased:
                FES = self.WTsim.getFreeEnergy() 
                FES -= FES.min()
                return FES / (self.kB*self.T_of_simulation)
            else: print('no FES in unbiased sim untill later analysis (external)')
        else: return self._FES

    def save_simulation_data_(self, folder_name=None):
        if folder_name is None: folder_name = ''
        else: folder_name = folder_name+'/'
        _time = '_'+str(self.simulation_timescale)+'ns'
        save_pickle_(self.xyz, folder_name+self.simulation_name+_time+'_xyz')
        save_pickle_(self.u,   folder_name+self.simulation_name+_time+'_u')
        save_pickle_(self.Tu,  folder_name+self.simulation_name+_time+'_Tu')
        if self.biased:
            save_pickle_(self.s,   folder_name+self.simulation_name+_time+'_s')
            save_pickle_(self.v,   folder_name+self.simulation_name+_time+'_v')
            save_pickle_(self.Tv,  folder_name+self.simulation_name+_time+'_Tv')
            save_pickle_(self.FES, folder_name+self.simulation_name+_time+'_FES(kT)')
        else: pass

    def load_simulation_data_(self, folder_name, n_frames=int(1e20), shuffle=False, _from=0):
        done = False
        if self.simulation_initialised:
            print('cannot load data when simulation_initialised, restart the whole object to import data')
        else:
            self._xyz = None
            self._u = None
            self._Tu = None
            self._s = None
            self._v = None
            self._vT = None
            self._FES = None
            files = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
            print('loading',files)
            for file in files:
                a = file.rfind('_')+1
                token = '_'+file[a:a+3]
                x = load_pickle_(folder_name+'/'+file)[_from:]
                if shuffle:
                    if not done:
                        inds_rand = np.random.choice(len(x),len(x),replace=False)
                        done = True
                    else: pass
                    x = x[inds_rand]
                else: pass
                setattr(self, token, x[:n_frames])  

def count_matrix_from_cluster_assigments_(c):
    ''' not used. 
    Input: c : from unbiased data
    Ouput: counts : square matrix : closer to symmetric -> unbiased sampling more likely to be ergodic
    '''
    n_states = len(set(c))
    c0 = c[0]
    counts = np.zeros([n_states,n_states]).astype(np.int32)
    for i in range(len(c)):
        if c[i] != c0:
            counts[c0,c[i]] += 1
            c0 = c[i]
        else: pass
    return counts

def fs_to_deltafs_(fs, ind_min:int=None, start=0, 
                                         end=int(1e20),
                    median=False):
    ''' gentle allignment of mutliple (converging) FE eastimates such that the lowest FE state (on average) is alligned to zero.
    Inputs:
        fs : (n_estimates, n_states) shaped array of (absolute) FE estimates
    Output:
        deltafs : = fs[i]+c[i] for some c[i] does not effect FE differences but helps to visualise FE differences.
    '''
    x = np.array(fs)
    x = np.where((x==np.inf)|(x==np.nan),0.0,x)
    x -= x.mean(1,keepdims=True)
    if ind_min is None:
        ind_min = np.argmin(x.mean(0))
        print('FE estimates of state',ind_min,'(often the lowest FE) moved to around 0')
    else: pass
    if median:
        deltafs = x - np.median(x[start:end,ind_min:ind_min+1])
    else:
        deltafs = x - x[start:end,ind_min:ind_min+1].mean()
    return deltafs

def deltau_states_in_time_(c, # (m,) from single simulation or multiple randomly initialised simulations
                           u, # (m,1) absolute potential energies from single simulation or multiple randomly initialised simulations
                           weights=None, # (m,1)
                           align = True,
                           ind_min = None,
                          ):
    ''' 
    Inputs:
        c : (m,) shaped array of integer cluster assigments of states during a simulation
        u : (m,1) shaped array of corresponding potential energies
        weights : (m,1) shaped array of weights if simulation was biased
    Output:
        (m,n_states) array of cumulative averages of potential energy (differences; if align) of/between states during the simulation
    '''
    # c : array of m integer cluster assigments
    c = np.array(c).flatten()
    m = len(c)
    if weights is None: weights = np.ones([m,1])
    else: pass
    oh = c_to_oh_(c) # (m,n_states) shaped one-hot encodings of states
    av_u_states = np.ma.divide(np.cumsum(oh*weights*u,axis=0),np.cumsum(oh*weights,axis=0))
    
    if align: return fs_to_deltafs_(av_u_states, ind_min=ind_min)
    else: return av_u_states

def c_to_oh_(c):
    # c : (m,) shaped array of integer cluster assigments
    # oh : (m,n_states) ; n_states = len(set(c) shaped array of one-hot encodings
    c = np.array(c).flatten()
    c -= c.min() # just incase.
    m = len(c)
    n_states = len(set(c))
    inds_states = [np.where(c==i)[0] for i in range(n_states)]
    oh = np.zeros([m,n_states])
    for i in range(n_states):
        oh[inds_states[i],i] = 1
    return oh

def deltaf_states_in_time_(c, # (m,) from single simulation or multiple randomly initialised simulations
                           weights=None, # (m,1) from single simulation or multiple randomly initialised simulations
                           align = True,
                           ind_min = None,
                          ):
    ''' similar to deltau_states_in_time_() but tracking the cumulative weights (uniformly weighted counts if weights is None).
    '''
    # c : array of m integer cluster assigments
    # output FE differences beteween states based on (weighted) counts
    c = np.array(c).flatten()
    m = len(c)
    if weights is None: weights = np.ones([m,1])
    else: pass
    oh = c_to_oh_(c)
    counts = np.cumsum(oh*weights[:len(oh)],axis=0)
    counts /= counts.sum(axis=1,keepdims=True)
    f_states = -np.log(counts)
    if align: return fs_to_deltafs_(f_states, ind_min=ind_min)
    else: return f_states

def block_average_(x, # one hot encodings only.
                   block_size:int,
                   weights = None,
                   FE = True,
                  ):
    ''' this was not used.
    '''
    # x : (m,...) such that x[i].sum()=1 for all i (i.e., one hot encodings)
    m = x.shape[0] # best if this is multiple of block size (warning added)
    shape = list(x.shape[1:])

    if weights is None: weights = np.ones([m,1])
    else:               weights = np.array(weights).reshape(m,1)
  
    n_blocks = m//block_size
    if m/block_size - n_blocks > 0: print('!! x.shape[0] is not multiple of block size')
    else: pass
    
    block_weights = np.zeros([n_blocks,1])
    block_averages = np.zeros([n_blocks]+shape)

    for i in range(n_blocks):
        a = i*block_size
        b = (i+1)*block_size
        
        x_i = x[a:b]
        ws_i = weights[a:b]
        block_weight = ws_i.sum()
        
        block_averages[i] = (x_i*ws_i).sum(0) / block_weight
        block_weights[i] = block_weight

    block_weights /= block_weights.sum()
    
    MU = (block_averages*block_weights).sum(0, keepdims=True)
    SE = np.sqrt((block_weights*(block_averages-MU)**2).sum(0))
    SE /= np.sqrt(n_blocks - 1)
    MU = MU[0]
    
    if FE:
        deltaf = -np.log(MU)
        deltaf_upper = -np.log(MU+SE)
        deltaf_lower = -np.log(MU-SE)
        C = - deltaf.min()
        deltaf += C
        deltaf_upper += C
        deltaf_lower += C
        return deltaf, deltaf_upper, deltaf_lower  # (shape,), (shape,), (shape,) in terms of FE (kT).
    
    else: 
        return MU, SE # (shape,), (shape,) in terms of probabilities.

##
def get_labels_from_CVs_alanine_dipeptide_(s):
    ''' gives 'cluster assignments' of states, given 2D CV (s) # shape of s: (m,2)
    '''
    m = s.shape[0] ; inds_all = set(np.arange(m).tolist())
        
    inds_s1A = set(np.where(np.logical_and(s[:,0]>0, s[:,0]<2))[0].tolist())
    inds_s1B = set(np.where(np.logical_and(s[:,0]>-2, s[:,0]<0))[0].tolist())
    inds_s1C = inds_all - inds_s1A - inds_s1B
    inds_s2A = set(np.where(np.logical_or(s[:,1]>1.5, s[:,1]<-2))[0].tolist())
    inds_s2B = inds_all - inds_s2A

    inds_IV = list(inds_all - inds_s1B - inds_s1C) # IV
    inds_III = list(inds_all - inds_s1A  - inds_s2A) # II
    inds_II = list(inds_all - inds_s1A  - inds_s1C - inds_s2B) # II
    inds_I = list(inds_all - inds_s1A - inds_s1B - inds_s2B) # I
    
    inds = [inds_I,inds_II,inds_III,inds_IV] ; n_states = len(inds)
     
    c = np.zeros([m,]).astype(np.int32) - 1
    for i in range(n_states):
        if len(inds[i])>0:
            c[inds[i]] = i
        else: pass
    return c # cluster assignmets 

def get_labels_from_CVs_ibuprofen_(s):
    ''' gives 'cluster assignments' of states, given 2D CV (s) # shape of s: (m,2)
    '''
    m = s.shape[0] ; inds_all = set(np.arange(m).tolist())

    inds_s1A = set(np.where(np.logical_and(s[:,0]>-2, s[:,0]<1))[0].tolist())
    inds_s1B = inds_all - inds_s1A
    inds_s2A = set(np.where(np.logical_and(s[:,1]>0, s[:,1]<2))[0].tolist())
    inds_s2B = set(np.where(np.logical_and(s[:,1]>-2, s[:,1]<0))[0].tolist())
    inds_s2C = inds_all - inds_s2A - inds_s2B

    inds_I = list(inds_all - inds_s2A - inds_s2B - inds_s1A)
    inds_II = list(inds_all - inds_s2A - inds_s2B - inds_s1B)
    inds_III = list(inds_all - inds_s2C - inds_s2B - inds_s1A)
    inds_IV = list(inds_all - inds_s2C - inds_s2A - inds_s1A)
    inds_V = list(inds_all - inds_s2C - inds_s2A - inds_s1B)
    inds_VI = list(inds_all - inds_s2C - inds_s2B - inds_s1B)

    inds = [inds_I, inds_II, inds_III, inds_VI, inds_IV, inds_V] ; n_states = len(inds)

    c = np.zeros([m,]).astype(np.int32) - 1
    for i in range(n_states):
        if len(inds[i])>0:
            c[inds[i]] = i
        else: pass
    return c # cluster assignmets 

def get_labels_from_CVs_ibuprofen_double_well_(s):
    ''' gives 'cluster assignments' of states, given 2D CV (s) # shape of s: (m,2)
    '''
    m = s.shape[0] ; inds_all = set(np.arange(m).tolist())
    
    inds_1 = np.where((s[:,1]<0)&(s[:,1]>-2))[0]
    inds_0 = np.array(list(inds_all - set(inds_1)))

    inds = [inds_0, inds_1] ; n_states = len(inds)

    c = np.zeros([m,]).astype(np.int32) - 1
    for i in range(n_states):
        if len(inds[i])>0:
            c[inds[i]] = i
        else: pass
    return c # cluster assignmets 

##

class Local_PE:
    ''' Eq 1. ( local potential energy u_{k} )
    '''
    def __init__(self,
                 global_potential_energy_function_,
                 clustering_function_, # clustering can be using any method in any dimensions as long as all decision boundaries remain fixed 
                 ):
        self.global_potential_energy_function_ = global_potential_energy_function_
        self.clustering_function_ = clustering_function_
        
    def inf_well_(self, xyz, k : int):
        m = xyz.shape[0]
        c = self.clustering_function_(xyz)
        zeros = np.zeros([m])
        u_well = np.where(c!=k,1e20,zeros)[:,np.newaxis]
        return u_well # potential energy of 'infinite square well' with 0 energy only if xyz in k'th metastable region of configurational space.
    
    def __call__(self,
                 xyz, # configuration
                 k = None, # index of which metastable state xyz is expected to belong to. If None; global potential energy u is evaluated.
                 use_this_T_instead=None,
                ):
        u = self.global_potential_energy_function_(xyz, T=use_this_T_instead)
        if k is not None:
             # reduced potential energy + penalty if not in state k
            return u + self.inf_well_(xyz,k)
        else:
             # reduced potential energy without any possible penalties
            return u

##

def split_into_training_data_(*inputs, labels, training_set_size, inds_rand=None):
    # shuffle all *inputs in time (along axis 0) in the same way and then split into separate arrays:
    # (states * training vs validation set) x len(inputs)
    # size of training set = size of validation set = same for all states (i.e., all states equally populated)
    '''
    *inputs : (m,...) MD arrays of interest, m = number of MD frames.
    labels : (m,) shaped array of associated cluster assignments (of metastable states).
    training_set_size : int <= m//2
    inds_rand : (m,) shaped array of unique indices 0,...,m-1 in random order [None or from previous save (for reproducibility)]
    '''
    m = len(inputs[0])
    if inds_rand is None: inds_rand = np.random.choice(m,m,replace=False)
    else: pass
    n_states = len(set(labels))
    labels = np.array(labels)[inds_rand]
    inds_states = [np.where(labels==k)[0] for k in range(n_states)]
    training = []
    validation = []
    n = training_set_size
    for x in inputs:
        x = np.array(x)[inds_rand]
        x_training = []
        x_validation = []
        for k in range(n_states):
            xk = x[inds_states[k]]
            x_training.append(xk[:n])
            x_validation.append(xk[n:2*n])
        training.append(x_training)
        validation.append(x_validation)
    return training, validation, inds_rand

##

def simple_smoothing_matrix_(S,N,c=1):
    # notperiodic.
    xs = np.linspace(0,1,S)
    z = np.linspace(0,1,N)
    c = -0.5/( (z[1]-z[0])/c )**2 ; W = []
    for x in xs:
        W.append(  np.exp(c*(z-x)**2)  )
    W = np.array(W)
    return W/W.sum(1)[:,np.newaxis]
    
def simple_smoother_(X,c=1.,S=None):
    ''' ~ given running average from stochastic data X : used only for visualisation.
    '''
    # for smoothing arrays(X) shaped as (N1,) or (N1,N2) or any (N1,N2,...,Nd) up to d=13.
    ' c: lim(c -> 0) -> ~ line of best fit '
    ' S: can be None (output same shape), or an int, or a list of ints (one for each axis of the input array). '
    Ns = X.shape ; dim = len(Ns)
    
    if S is None: Ss = Ns
    if type(S) is int: Ss = tuple([S]*dim)
    else: pass
        
    summation_indices = 'ijklmnopqrstu'
    output_indices = 'abcdefghvwxyz'
    
    einsum_args = []
    string = ''

    for ax in range(dim):
        einsum_args.append(simple_smoothing_matrix_(Ss[ax],Ns[ax],c=c))
        string += (output_indices[ax] + summation_indices[ax] + ',')
    
    string += (summation_indices[:dim] + '->' + output_indices[:dim]) 
    einsum_args.insert(0,string)
    einsum_args.append(X)
    
    return np.einsum(*einsum_args)

##

def pool_(x, ws=None):
    ''' Output: weighted average of x '''
    if ws is None: return x.mean()
    else:          return (x*ws).sum() / ws.sum()
    
def get_phi_ij_(models, data, potential_energy_function, evalation_sample_size=10000, shuffle=True):
    ''' targeted MBAR:
    Inputs:
        models : list of K initialised, trained models which have methods called forward_graph_ and inverse_graph_
        data : list of K (m,n_atoms,3) shaped arrays of conformers sampled by locally ergodic MD from states 0,...,K-1 respectively.
        potential_energy_function : instance of Local_PE
        evalation_sample_size : int = n <= m
        shuffle : if False first m conformers from each 'data' array are used, else random m conformers are drawn from these arrays.
    Output:
        (K,K,evalation_sample_size) shaped array of remapped potential energies = input to pymbar to obtain FE estimates.
    '''
    n_states = len(data)
    # no shuffling here, it doesnt matter.
    phi_ij = np.zeros([n_states,n_states,evalation_sample_size])
    
    for i in range(n_states):

        ri = data[i]

        if shuffle: ri = np.array(ri[np.random.choice(len(ri),evalation_sample_size,replace=False)])
        else:       ri = np.array(ri[:evalation_sample_size])

        z, ladJi = models[i].forward_graph_( np2tf_(ri) )

        for j in range(n_states):

            r, ladJj = models[j].inverse_graph_(z)

            phi_ij[i,j] = potential_energy_function(tf2np_(r), k=j)[:,0] - tf2np_(ladJi + ladJj)[:,0]
            
    return np.einsum('ijk->jik',phi_ij) # (n_states,n_states,n)

sigma = lambda x : (1.0+np.exp(-x))**(-1)

def BAR_(incuA, incuB, wsA=None, wsB=None, f_window_grain = [-10,10,1000]):
    ''' local implementation of 2-state BAR, finding f (via grid search) that best fits the BAR equality A = B below.
    Inputs as used in current work:
        incuA : (m,1) shaped array = \phi(r) = u(r) + ln(p(r)) ; r ~ \mu
            \mu = \mu(r) = exp(-u(r)) / Z is underlying NVT distribution of MD data with 
                unknown normalisation constant Z, but we have m ergodic samples r ~ \mu
                Want output f to be as close as possible to underlying -ln(Z).

        incuB : (m,1) shaped array = \phi(r) = u(r) + ln(p(r)) ; r ~ p
            p is some analytical normalised distribution (e.g., flow based generative model) which is similar to \mu,
            We can evaluate ln(p) and can sample m ergodic samples from it r ~ p

        wsA : is the sampling from \mu ergodic? If not then set wsA = known weights, else None.
        wsB : is the sampling from   p ergodic? Yes, by default (wsB = None).
        f_window_grain : list [float, float, int] = [a,b,grain]
            a = minimum f that may be valid
            b = maximum f that may be valid
            grain = how many values on a grid to try between a and b
    Output:
        f : scalar : estimate of -ln(Z)
    '''
    a, b, grain = f_window_grain
    grid_f =  np.linspace(a, b, grain)
    errs = []
    for f in grid_f:
        A = pool_(sigma( (f - incuA) ), ws=wsA)
        B = pool_(sigma( (incuB  - f) ), ws=wsB)
        errs.append(np.abs(A-B))
    ind_min = np.argmin(errs)
    f = grid_f[ind_min]
    if ind_min in [0,grain-1]:
        print('!! warning: BAR_ : grid not adjusted properly')
    else: pass
    return f

def get_FE_estimates_(model,              # model p_{k} similar to \mu_{k}, because pretrained on data from \mu_{k}
                      u_function,         # = lambda r : u_(r, k = k ) ; k = index of metastable state of interest.
                      xyz_u_train : list, # [MD data from from state k., corresponding reduced potential energies]
                      xyz_u_val : list,   # [different MD data from from state k., corresponding reduced potential energies]
                      MD_weights_train = None, # if the MD data not ergodic, reweighting weights to correct for this nonergodicity.
                      MD_weights_val = None,   # if the different MD data not ergodic, reweighting weights to correct for this nonergodicity.
                      evalation_sample_size = 10000, # int : higher is better (default to 10,000).
                      shuffle = True,                # does not matter too much, True is better.
                      temperature_already_added = None,  # dont use.
                      temperature_try = None,            # dont use.
                      f_window_grain = [-10,10,2000],    # see BAR_()
                      name_save_BAR_inputs : str = None, # file name where pymbar input arrays shall be saved for 2state BAR estimator.
                      ):
    ''' all estimators of absolute FE (f) of a state with index k.
    '''
    xyz_T, u_T = xyz_u_train
    xyz_V, u_V = xyz_u_val
    m_T = len(xyz_T) ; m_V = len(xyz_V)
    if evalation_sample_size > m_T:
        print('!!! get_FE_estimates_ : evalation_sample_size:',
              evalation_sample_size,'> training_dataset_size:',m_T)
    else: pass
    if evalation_sample_size > m_V:
        print('!!! get_FE_estimates_ : evalation_sample_size:',
              evalation_sample_size,'> validation_dataset_size:',m_V)
    else: pass
    if MD_weights_train is None: w_T = np.ones([m_T,1])
    else: w_T = MD_weights_train
    if MD_weights_val is None: w_V = np.ones([m_V,1])
    else: w_V = MD_weights_val
    if shuffle:
        inds_rand_T = np.random.choice(m_T, evalation_sample_size, replace=False)
        xyz_T = np.array(xyz_T[inds_rand_T])
        u_T = np.array(u_T[inds_rand_T])
        w_T = np.array(w_T[inds_rand_T])
        inds_rand_V = np.random.choice(m_V, evalation_sample_size, replace=False)
        xyz_V = np.array(xyz_V[inds_rand_V])
        u_V = np.array(u_V[inds_rand_V])
        w_V = np.array(w_V[inds_rand_V])
    else:
        xyz_T = np.array(xyz_T[:evalation_sample_size])
        u_T = np.array(u_T[:evalation_sample_size])
        w_T = np.array(w_T[:evalation_sample_size])
        xyz_V = np.array(xyz_V[:evalation_sample_size])
        u_V = np.array(u_V[:evalation_sample_size])
        w_V = np.array(w_V[:evalation_sample_size])         

    if temperature_already_added is None: temperature_already_added = 1.0
    else: pass
    if temperature_try is None: temperature_try = 1.0
    else: pass
    T_rescale = temperature_already_added/temperature_try

    ##

    negS_T = model.ln_model(xyz_T)[-1]
    f_T = u_T*T_rescale + negS_T
    offset = np.median(f_T)
    f_T -= offset

    negS_V = model.ln_model(xyz_V)[-1]
    f_V = u_V*T_rescale + negS_V - offset

    r_samples, negS_BG = model.sample_model(evalation_sample_size)
    u_BG = u_function(r_samples)
    f_BG = u_BG*T_rescale + negS_BG - offset

    #
    AVMD_T = pool_(f_T,ws=w_T) + offset
    AVMD_V = pool_(f_V,ws=w_V) + offset
    AVBG   = pool_(f_BG,ws=None) + offset

    EXPMD_T = np.log(pool_(np.exp(f_T),ws=w_T)) + offset
    EXPMD_V = np.log(pool_(np.exp(f_V),ws=w_V)) + offset
    EXPBG   =-np.log(pool_(np.exp(-f_BG),ws=None)) + offset

    BAR_T = BAR_(incuA=f_T, incuB=f_BG, wsA=w_T, f_window_grain = f_window_grain) + offset
    BAR_V = BAR_(incuA=f_V, incuB=f_BG, wsA=w_V, f_window_grain = f_window_grain) + offset
    
    if name_save_BAR_inputs is not None:
        # !! ignoring T_rescale

        first_row = np.concatenate([u_T, u_BG], axis=0)
        second_row = np.concatenate([u_T-(f_T+offset), u_BG-(f_BG+offset)], axis=0)
        mBAR_inputs_T = np.stack([first_row, second_row],axis=0)[:,:,0]
        save_pickle_(mBAR_inputs_T, name_save_BAR_inputs + '_T')

        first_row = np.concatenate([u_V, u_BG], axis=0)
        second_row = np.concatenate([u_V-(f_V+offset), u_BG-(f_BG+offset)], axis=0)
        mBAR_inputs_V = np.stack([first_row, second_row],axis=0)[:,:,0]
        save_pickle_(mBAR_inputs_V, name_save_BAR_inputs + '_V')

    else: pass
    
    ##

    #'''
    _BAR_T = BAR_(incuA=f_T, incuB=f_BG, f_window_grain = f_window_grain) + offset # ignoring weights (SI fig)
    _BAR_V = BAR_(incuA=f_V, incuB=f_BG, f_window_grain = f_window_grain) + offset # ignoring weights (SI fig)
    #'''
    '''
    offset_u_T = np.median(u_T)
    BAR_u_T =  BAR_(incuA=u_T-offset_u_T, incuB=u_BG-offset_u_T, wsA=w_T, f_window_grain = f_window_grain) + offset_u_T
    offset_u_V = np.median(u_V)
    BAR_u_V =  BAR_(incuA=u_V-offset_u_V, incuB=u_BG-offset_u_V, wsA=w_V, f_window_grain = f_window_grain) + offset_u_V
    '''
    ##
    return np.array([AVMD_T,  # 0
                     AVMD_V,  # 1
                     AVBG,    # 2
                     EXPMD_T, # 3
                     EXPMD_V, # 4
                     EXPBG,   # 5
                     BAR_T,   # 6
                     BAR_V,   # 7
                     u_T.mean(), # 8 #-5
                     u_V.mean(), # 9 # -4
                     u_BG.mean(), # 10 # -3
                     #BAR_u_T, # 11 # -2
                     #BAR_u_V, # 12 # -1
                     _BAR_T,
                     _BAR_V,
                    ])

class TRAINER:
    '''
    for training a set of separate models on separate datasets and roughly plotting FE estimates during training, saving all arrays underlying these estimates and arrays needed for MBAR.
    '''
    def __init__(self,
                 models : list, # list of separate instances of MODEL_3
                 max_training_batches : int, # some high number : maximum number of expected training batches
                 n_batches_between_evaluations = 50, # stride of how many batches to skip beetween (FE evaluations and saving the arrays)
                 ):
        self.n_main_estimates = 8+3+2
        self.models = models
        self.max_training_batches = max_training_batches
        self.n_batches_between_evaluations = n_batches_between_evaluations

        self.n_states = len(models)
        self._AVMD_f_T_all = np.zeros([self.max_training_batches, self.n_states])
        self._AVMD_S_T_all = np.zeros([self.max_training_batches, self.n_states])

        self._evaluation_grid = np.arange(self.n_batches_between_evaluations,
                                          self.max_training_batches+self.n_batches_between_evaluations,
                                          self.n_batches_between_evaluations)-1
        self._estimates = np.zeros([self.n_states,len(self._evaluation_grid),self.n_main_estimates])
        self.count = 0
        self.count_strided = 0

    # inputs to train_ seen from notebooks
    def train(self,
              n_batches : int, # < max_training_batches 

              xyz_training,
              u_training,

              xyz_validation,
              u_validation,

              potential_energy_function,

              w_training = None,
              w_validation = None,

              training_batch_size = 1000,
              evalation_batch_size = 10000,

              evaluate_main = True,
              name_save_BAR_inputs = None,
              name_save_mBAR_inputs = None,

              shuffle = True,
              f_window_grain_BAR_local =  [-50,50,2000],
              delta_f_ground_truth = None,
              ):
        if w_training is None:  w_training = [None]*self.n_states
        else: pass
        if w_validation is None:  w_validation = [None]*self.n_states
        else: pass

        for i in range(n_batches):
            AVMD_f_T_i = np.zeros([self.n_states,])
            AVMD_S_T_i = np.zeros([self.n_states,])
            for k in range(self.n_states):
                AVMD_f_T_ik, AVMD_S_T_ik = self.models[k].step_ML_(
                xyz = xyz_training[k],
                u = u_training[k],
                batch_size = training_batch_size,
                                                                  )
                AVMD_f_T_i[k] = AVMD_f_T_ik
                AVMD_S_T_i[k] = AVMD_S_T_ik

            self._AVMD_f_T_all[self.count] = AVMD_f_T_i
            self._AVMD_S_T_all[self.count] = AVMD_S_T_i

            print(str(i)+'_AVMD_f_T_'+str(self.count)+':', AVMD_f_T_i)

            if self.count in self._evaluation_grid:
                if name_save_mBAR_inputs is None:
                    pass
                else:
                    PHIij_T = get_phi_ij_(models=self.models,
                                          data = xyz_training,
                                          potential_energy_function = potential_energy_function,
                                          evalation_sample_size = evalation_batch_size,
                                          shuffle = shuffle)
                    PHIij_V = get_phi_ij_(models=self.models,
                                          data = xyz_validation,
                                          potential_energy_function = potential_energy_function,
                                          evalation_sample_size = evalation_batch_size,
                                          shuffle = shuffle)
                    save_pickle_(PHIij_T, name_save_mBAR_inputs+'_PHIij_T_'+str(self.count_strided))
                    save_pickle_(PHIij_V, name_save_mBAR_inputs+'_PHIij_V_'+str(self.count_strided))

                if evaluate_main:
                    for k in range(self.n_states):
                        if name_save_BAR_inputs is None: name_BAR = None
                        else: name_BAR = name_save_BAR_inputs+'_BAR_input_'+str(self.count_strided)+'_state'+str(k)+'_'
                        estimates_kc = get_FE_estimates_(model = self.models[k],
                                                        u_function  = lambda r : potential_energy_function(r, k=k),
                                                        xyz_u_train = [xyz_training[k], u_training[k]],
                                                        xyz_u_val   = [xyz_validation[k], u_validation[k]],
                                                        MD_weights_train = w_training[k],
                                                        MD_weights_val = w_validation[k],
                                                        evalation_sample_size = evalation_batch_size,
                                                        shuffle = shuffle,
                                                        name_save_BAR_inputs = name_BAR,
                                                        f_window_grain = f_window_grain_BAR_local)
                        self._estimates[k,self.count_strided] = estimates_kc
                else: pass
                self.count_strided += 1
                print ("\033[A                             \033[A")
            self.count += 1

            if delta_f_ground_truth is not None and self.count_strided > 0:
                'verbose'
                clear_output(wait=True)
                [plt.plot([-1,self.evaluation_grid[-1]],[delta_f_ground_truth[k]]*2, color = 'C'+str(k), linewidth=5) for k in range(self.n_states)]
                es = self.estimates

                which = 0
                if self.n_states > 1:
                    show = fs_to_deltafs_(es[:,:,which].T, ind_min=None, start=None)
                else: 
                    show = np.array(es[:,:,which].T)
                plt.plot(self.evaluation_grid, show, alpha=0.5, color='grey', linewidth=1)
                which = 1
                if self.n_states > 1:
                    show = fs_to_deltafs_(es[:,:,which].T, ind_min=None, start=None)
                else:
                    show = np.array(es[:,:,which].T)
                plt.plot(self.evaluation_grid, show, alpha=0.5, color='blue', linewidth=2)

                which = 6
                if self.n_states > 1:
                    show = fs_to_deltafs_(es[:,:,which].T, ind_min=None, start=None)
                else:
                    show = np.array(es[:,:,which].T)
                plt.plot(self.evaluation_grid, show, alpha=0.5, color='orange', linewidth=3)
                which = 7
                if self.n_states > 1:
                    show = fs_to_deltafs_(es[:,:,which].T, ind_min=None, start=None)
                else:
                    show = np.array(es[:,:,which].T)
                plt.plot(self.evaluation_grid, show, alpha=0.5, color='green', linewidth=3)

                plt.ylim(-0.5,delta_f_ground_truth.max()+0.5)
                plt.xlim(-1,self.evaluation_grid[-1]+1)
                plt.show()
                
            else: pass
            
    @property
    def estimates(self,):
        return np.array(self._estimates[:,:self.count_strided])
    
    @property
    def evaluation_grid(self,):
        return np.array(self._evaluation_grid[:self.count_strided])
    
    @property
    def AVMD_f_T_all(self,):
        return np.array(self._AVMD_f_T_all[:self.count])
    
    @property
    def AVMD_S_T_all(self,):
        return np.array(self._AVMD_S_T_all[:self.count])
    
    def save_the_above_(self, name : str):
        save_pickle_([self.estimates,
                      self.evaluation_grid,
                      self.AVMD_f_T_all,
                      self.AVMD_S_T_all
                      ], name)

##

def save_coordiantes_as_pdb_(coordinates, name, la=None, verbose=False):
    ''' to quickly visualise 3D points in VMD:
    Inputs:
        coordinates : (n_frames,n_atoms*3) shaped array
        name : str : 'this_file_name'
        la : atom labels : (n_frames,n_atoms) shaped array or None : numbers for b-factor column
    '''                   
    coordinates = np.array(coordinates).round(decimals=2)
    n_frames = coordinates.shape[0]
    if la is None: values = np.zeros((n_frames,int(coordinates.shape[1]/3)))
    else: values = la
    pdb = open(name+'.pdb', "w")
    spaces = {0: "", 1: " ", 2: "  ", 3: "   ", 4: "    ", 5: "     ", 6: "      "}
    zeros =  {0: '', 1: '0', 2: '00', 3: '000', 4: '0000', 5: '00000'}
    frame = 0
    for i in range(0, n_frames):
        atom_index = 0
        stride = 0
        for j in range(0, int(len(coordinates[frame]) / 3)):
            atom_index = atom_index + 1
            pdb_row = []
            x_index = int(stride) ; stride = stride + 1
            y_index = int(stride) ; stride = stride + 1
            z_index = int(stride) ; stride = stride + 1
            x = str(coordinates[frame, x_index])
            y = str(coordinates[frame, y_index])
            z = str(coordinates[frame, z_index])
            n_spaces_to_add = 7 - len(str(atom_index))
            I = ("ATOM", spaces[n_spaces_to_add], str(atom_index), "  C   HET X")
            n_spaces_to_add = 4 - len(str(atom_index))
            II = (spaces[n_spaces_to_add], str(atom_index), "      ")
            n_0s_to_add = 6 - len(x)
            III = (x, zeros[n_0s_to_add], "  ")
            n_0s_to_add = 6 - len(y)
            IV = (y, zeros[n_0s_to_add], "  ")
            n_0s_to_add = 6 - len(z)
            if values[i,j] <0: sign = '-'
            else:              sign = ' '
            V = (z, zeros[n_0s_to_add], " "+sign+str(np.abs(float(values[i,j])).round(2))+" "+sign+str(np.abs(float(values[i,j])))+"           C")
            pdb_row.append(''.join(I))
            pdb_row.append(''.join(II))
            pdb_row.append(''.join(III))
            pdb_row.append(''.join(IV))
            pdb_row.append(''.join(V))
            pdb.write(''.join(pdb_row) + "\n")
        pdb.write("END" + "\n") # pdb.write("ENDMOL" + "\n")
        frame = frame + 1
        if verbose: print("new frame, "+str(i))
        else: pass
    pdb.close()
    print("saved",name+'.pdb')
##

class XR_MAP_toy:
    
    def __init__(self,
                 raw_data : np.ndarray, centre=False):
        
        if len(raw_data.shape) > 2: print('wrong shape of data provided')
        else: pass
        
        self.dim = raw_data.shape[1]
        self.data_ranges = [[raw_data[:,i].min()-1e-3, raw_data[:,i].max()+1e-3] for i in range(self.dim)]
        
        self.model_range = [-1.0, 1.0]
        
        if centre:
            self.mean = 0.0
            X = self.forward(raw_data)[0]
            self.mean = tf.reduce_mean(X,axis=0,keepdims=True)
        else:
            self.mean = 0.0
        
    def fit_to_range_(self, x, physical_range : list, forward : bool = True):
        
        # shape: x ~ (m,n) ; m = batch_size , n = number of alike variables. 
        
        # forward : physical_range -> model_range
        # else    : model_range -> physical_range
        
        x_min, x_max = physical_range
        min_model, max_model = self.model_range
        
        J = (max_model - min_model)/(x_max - x_min)
        
        if forward:
            return J*(x - x_min) + min_model , tf.cast(tf.math.log(J)*x.shape[1], tf.float32)
        else:
            return (x - min_model)/J + x_min , tf.cast(-tf.math.log(J)*x.shape[1], tf.float32)

    def forward(self, R):

        if len(R.shape) < 2: R = R.reshape(len(R), self.dim)
        else: pass

        ladJrx = 0
        
        X = []
        for i in range(self.dim):
            x, ladJ = self.fit_to_range_(R[:,i:i+1], physical_range=self.data_ranges[i], forward=True)
            X.append(x)
            ladJrx += ladJ
        
        X = tf.concat(X, axis=1) - self.mean
        return X, ladJrx

    def inverse(self, X):

        if len(X.shape) < 2: X = X.reshape(len(X), self.dim)
        else: pass
        
        X += self.mean

        ladJxr = 0
        
        R = []
        for i in range(self.dim):
            r, ladJ = self.fit_to_range_(X[:,i:i+1], physical_range=self.data_ranges[i],forward=False)
            R.append(r)
            ladJxr += ladJ
        
        R = tf.concat(R, axis=1)
        return R, ladJxr

##

point_ = lambda x, p, s:      1.0 + tf.reduce_sum(((x-p)/s)**2, axis=1, keepdims=True) # p, \sigma
barrier_ = lambda x, mu, s, h:  h*tf.exp(  -0.5 * tf.reduce_sum(((x-mu))**2/s, axis=1, keepdims=True)  ) 
minimum_ = lambda x, c, R, s, n: tf.reduce_sum( (tf.einsum('ij,jk->ik',x-c,R)*(1/s))**n  , axis=1, keepdims=True)

class TOY_POTENTIAL_FH:
    # 3D potential energy surface from https://doi.org/10.1021/acs.jctc.9b00907
    def __init__(self,
                 p_points, # (P,D)
                 s_points, # (P,D)
                 
                 mu_barriers, # (B,D)
                 s_barriers, # (B,D)
                 h_barriers, # (B,1)
                 
                 c_minima, # (C,D)
                 R_minima, # (C,D,D)
                 s_minima, # (C,D)
                 n_minima, # (C,D)
                 
                 alpha, # (,)
                ):
        
        self.D = p_points[0].shape[0]
        
        self.P = len(p_points)
        self.p_points = [np2tf_(x[np.newaxis,:]) for x in p_points]
        self.s_points = [np2tf_(x[np.newaxis,:]) for x in s_points]
        
        self.B = len(h_barriers)
        self.mu_barriers = [np2tf_(x[np.newaxis,:]) for x in mu_barriers]
        self.s_barriers = [np2tf_(x[np.newaxis,:]) for x in s_barriers]
        self.h_barriers = [np2tf_(x) for x in h_barriers]
        
        self.C = len(c_minima)
        self.c_minima = [np2tf_(x[np.newaxis,:]) for x in c_minima]
        self.R_minima = [np2tf_(x) for x in R_minima]
        self.s_minima = [np2tf_(x[np.newaxis,:]) for x in s_minima]
        self.n_minima = [np2tf_(x[np.newaxis,:]) for x in n_minima]
        
        self.alpha = alpha
        
    def U_(self, x):
        # x : (m,D)
        
        P_sum, B_sum, C_sum = 0.0, 0.0, 0.0
        
        for i in range(self.P):
            P_sum += 1/point_(x, self.p_points[i], self.s_points[i])
            
        for i in range(self.B):
            B_sum += barrier_(x, self.mu_barriers[i], self.s_barriers[i], self.h_barriers[i])
            
        for i in range(self.C):
            C_sum += 1/minimum_(x, self.c_minima[i], self.R_minima[i], self.s_minima[i], self.n_minima[i])
            
        energies = alpha * ( 1.0/(C_sum + P_sum) + B_sum)
        
        return energies # (m,1)
        
    def evaluate_potential(self, x, return_grad=False, numpy=False):
        if numpy: x = np2tf_(x)
        else: pass
        
        if return_grad:
            with tf.GradientTape() as tape:
                tape.watch(x)
                energies = self.U_(x)
                Energies = energies[:,0]
            forces = tape.gradient(Energies, x)
            if numpy:
                return tf2np_(energies), tf2np_(forces)
            else:
                return energies, forces
        else:
            energies = self.U_(x)
            if numpy:
                return tf2np_(energies)
            else:
                return energies
        
## 3D:
#'''
p_points =      [np.array([-1,-1,-1]),
                 np.array([-1,-1,1]),
                 np.array([1,-1,1]),
                 np.array([1,1,-1]),
                 np.array([-1,1,1]),
                 np.array([1,-1,-1])
                ]
s_points =      [np.array([0.2,0.2,0.2]) for i in range(6)]

mu_barriers =   p_points
s_barriers =    s_points 
h_barriers =    [1 for i in range(6)]

c_minima = [np.array([0,-1,-1]),
                 np.array([-1,-1,0]),
                 np.array([0,-1,1]),
                 np.array([1,0,0]),
                 np.array([0,1,0]),
                 np.array([0,0,0]),
                ]
C = (2**0.5)/2

R_minima = [
    
                 np.eye(3),
                 np.eye(3),
                 np.eye(3),
    
                np.array([[1,0,0],
                          [0,C,C],
                          [0,-C,C]]),
                 
                 np.array([[C,0,C],
                           [0,1,0],
                           [-C,0,C]]),
                 
                 np.array([[-C,-C,0],
                           [0.5,-0.5,-C],
                           [0.5,-0.5,C]]).T,
                ]
s_minima = [np.array([1.0,0.1,0.1]),
                 np.array([0.1,0.1,1.0]),
                 np.array([1.0,0.1,0.1]),
                 np.array([0.1,2**0.5,0.1]),
                 np.array([2**0.5,0.1,0.1]),
                 np.array([0.1,3**0.5,0.1])
                ]
n_minima = [np.array([8,2,2]),
                 np.array([2,2,8]),
                 np.array([8,2,2]),
                 np.array([2,8,2]),
                 np.array([8,2,2]),
                 np.array([2,8,2]),
                 ]

alpha = 30.0
#'''
def set_default_reduced_toy_potential_():
    toy_U = TOY_POTENTIAL_FH(
                             p_points = p_points, # (P,D)
                             s_points = s_points, # (P,D)

                             mu_barriers = mu_barriers, # (B,D)
                             s_barriers = s_barriers, # (B,D)
                             h_barriers = h_barriers, # (B,1)

                             c_minima = c_minima, # (C,D)
                             R_minima = R_minima, # (C,D,D)
                             s_minima = s_minima, # (C,D)
                             n_minima = n_minima, # (C,D)

                             alpha = alpha, # (,)
                            )
    toy_potential_ = lambda x : toy_U.evaluate_potential(x, return_grad=False, numpy=True)

    def reduced_toy_potential_(x, T=None):
        __kB = 1.0
        U = toy_potential_(x)
        if T is None: return U/(__kB*2.0)
        else:         return U/(__kB*T)

    return reduced_toy_potential_

def get_1modal_labels_toy_example_(x, states = None):
    c = 0.75
    _0 = np.where(x[:,0]>c)[0].tolist()
    _1 = np.where(x[:,1]>c)[0].tolist()
    _2 = np.where(np.abs(x).max(-1) < 0.9)[0].tolist()
    _3 = np.where((x[:,1]<-c)&(x[:,2]<-c))[0].tolist()
    _4 = np.where((x[:,0]<-c)&(x[:,1]<-c))[0].tolist()
    _5 = np.where((x[:,1]<-c)&(x[:,2]>c))[0].tolist()

    inds = [_0,_1,_2,_3,_4,_5]
    
    if states is not None:
        inds = [inds[states[i][0]]+inds[states[i][1]] for i in range(len(states))]
    else: pass
    
    c = np.zeros([len(x),]).astype(np.int32) - 1
    for i in range(len(inds)):
        c[inds[i]] = i
    
    return c

get_2modal_labels_toy_example_ = lambda x : get_1modal_labels_toy_example_(x, states=[[3, 5], [0, 4], [2, 1]])

get_no_labels_toy_example_ = lambda x : np.zeros([len(x),]).astype(np.int32)

##

def plot_points_(X,
                 la=None,
                 s=10,cmap='jet',
                 show_axes=True,
                 figsize=(5, 3),
                 autoscale=True,
                 show_colorbar=True,
                 **kwargs):
    """ Plot 2D or 3D points. 
        X : (m,d) shaped array of m points. [d = 2 or 3] 
        la = (m,) shaped array of m labels, for colour.
    """
    d = X.shape[1]
    
    if la is None: la = np.arange(len(X))
    else: pass 
    
    fig = plt.figure(figsize=figsize)
    if d >= 3: ax = fig.add_subplot(111, projection='3d') ; ax.set_zlabel('z')
    elif d == 2: ax = fig.add_subplot(111)
    else: pass
    
    if autoscale: pass 
    else: ax.autoscale(enable=None, axis='both', tight=False)
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if not show_axes: ax.set_axis_off()
    else: pass
    if d >= 3: img = ax.scatter(X[:,0],X[:,1],X[:,2],s=s,c=la,cmap=cmap,**kwargs)
    elif d == 2: img = ax.scatter(X[:,0],X[:,1],s=s,c=la,cmap=cmap,**kwargs)
    else: pass
    if show_colorbar: fig.colorbar(img)
    else: pass
    plt.show()
