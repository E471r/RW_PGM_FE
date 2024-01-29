from utils import *
from model import *

##

model = MODEL_3.load_model(saved_models_dir+'model_name')

r = model.sample_model(10000)[0] # (m,n_atoms,3)

save_coordiantes_as_pdb_(r.reshape([10000,n_atoms*3])*10.,'conformers_from_model_name')

# DynamicBonds in VMD