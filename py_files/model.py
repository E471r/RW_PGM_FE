import numpy as np

import tensorflow as tf

from spline_layer import SPLINE_COUPLING_LAYER

import pickle

##

def save_pickle_(x,name):
    with open(name, "wb") as f: pickle.dump(x, f) ; print('saved',name)
    
def load_pickle_(name):
    with open(name, "rb") as f: x = pickle.load(f) ; return x

##

DTYPE_tf = tf.float32
tf2np_ = lambda x : x.numpy() 
np2tf_ = lambda x : tf.cast(x, dtype=tf.float32)

##

class ic_map_identity:
    def __init__(self,):
        ''
    def forward(self,x):
        return x, 0.0
    def inverse(self,x):
        return x, 0.0

def get_list_cond_masks_unsupervised_(dim_flow):
    """ Reference: TABLE 1 in arXiv:2001.05486v2 (i-flow)
    """
    list_cond_masks = []
    for i in range(dim_flow):
        a = 2**i
        x = np.array((([0]*a + [1]*a)*dim_flow)[:dim_flow])
        if 1 in x: pass
        else: break
        list_cond_masks.append(x)
    return list_cond_masks # plt.matshow(list_cond_masks)

def get_random_flow_masks_(dim_flow, n_layers, include_B=False):
    masks = []
    for i in range(n_layers):
        A = np.where(np.random.rand(dim_flow)>0.5,1,0)
        if not include_B: masks += [A.tolist()]
        else:             masks += [A.tolist(), (1 - A).tolist()]
    return masks

def batched_evaluation_(_function, _input, batch_size, function_output_dims:list):
    # useful to save memory in some cases.
    # function_output_dims : list of lists of ints
    N = _input.shape[0]
    block_size = min(batch_size,N)
    n_blocks = N//block_size
    n_remainder = N%block_size

    n_outputs = len(function_output_dims)
    if n_outputs == 1: f = lambda x: [_function(x)]
    else: f = _function
    for i in range(n_outputs):
        locals()['output_'+str(i)] = np.zeros([N]+function_output_dims[i])

    for j in range(n_blocks):
        a = j*block_size
        b = (j + 1)*block_size
        outputs_j = f(_input[a:b])
        for i in range(n_outputs):
            locals()['output_'+str(i)][a:b] = outputs_j[i]

    if n_remainder == 0: pass
    else:
        outputs_remainder = f(_input[b:])
        for i in range(n_outputs):
            locals()['output_'+str(i)][b:] = outputs_remainder[i]

    y = [] # final outputs
    for i in range(n_outputs):
        y.append(locals()['output_'+str(i)])

    return y

class MODEL_3(tf.keras.models.Model):
    def __init__(self,
                 periodic_mask, # list with shape (dim_flow,) ; elements ints either 0 or 1. [can be all 0 or all 1]
                 list_cond_masks = None, # list [list with shape (dim_flow,), list with shape (dim_flow,), ... ]
                    # length of this list determines number of layers.
                    # all elements ints either 0 or 1, and not all 1 or 0 in each inner list.

                 IC_map = None,
                 optimiser_LR_decay = [0.001,0.0001], # [learning_rate, rate_decay]

                 n_bins_periodic = 8,            # same in each layer
                 number_of_splines_periodic = 1, # same in each layer
                 n_bins_other = 8,               # same in each layer

                 n_hidden = 1,
                 hidden_activation = tf.nn.silu,

                 min_bin_width = 0.001,
                 trainable_slopes = True,
                 min_knot_slope = 0.001,

                 dims_hidden = None,

                 nk_for_periodic_MLP_encoding = 1,

                 verbose : bool = True,
                 ):
        super(MODEL_3, self).__init__()
        ''' the least tidy part.
        '''
        self.init_args = [periodic_mask,
                          list_cond_masks,
                          IC_map,
                          optimiser_LR_decay,
                          n_bins_periodic,
                          number_of_splines_periodic,
                          n_bins_other,
                          n_hidden,
                          hidden_activation,
                          min_bin_width,
                          trainable_slopes,
                          min_knot_slope,
                          dims_hidden,
                          nk_for_periodic_MLP_encoding,
                          verbose,
                         ]

        periodic_mask = np.array(periodic_mask).flatten()
        self.dim_flow = len(periodic_mask)
        if list_cond_masks is not None: list_cond_masks = [np.array(x).flatten() for x in list_cond_masks]
        else: list_cond_masks = get_list_cond_masks_unsupervised_(self.dim_flow)
        
        self.periodic_mask = periodic_mask
        self.list_cond_masks = list_cond_masks
        self.n_bins_periodic = n_bins_periodic
        self.number_of_splines_periodic = number_of_splines_periodic
        self.n_bins_other = n_bins_other
        self.n_hidden = n_hidden
        self.hidden_activation = hidden_activation
        self.flow_range = [-1.0,1.0] # fixed
        self.min_bin_width = min_bin_width
        self.trainable_slopes = trainable_slopes # fixed to True
        self.min_knot_slope = min_knot_slope
        self.dims_hidden = dims_hidden
        self.nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding 
        self.verbose = verbose

        self.n_layers = len(list_cond_masks)
        self.inds_layers_forward = np.arange(self.n_layers)
        self.inds_layers_inverse = np.flip(self.inds_layers_forward)

        # list of spline-coupling-layers = self.LAYERS = trainable bijector.
        self.LAYERS = [SPLINE_COUPLING_LAYER(   periodic_mask = periodic_mask, 
                                                cond_mask = x, 
                                                n_bins_periodic = n_bins_periodic,
                                                number_of_splines_periodic = number_of_splines_periodic,
                                                n_bins_other = n_bins_other,
                                                n_hidden = n_hidden,
                                                hidden_activation = hidden_activation,
                                                flow_range = self.flow_range,
                                                min_bin_width = min_bin_width,
                                                trainable_slopes = trainable_slopes,
                                                min_knot_slope =  min_knot_slope,
                                                dims_hidden = dims_hidden,
                                                nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding)
                        for x in list_cond_masks
                        ]
        
        self.IC_map = ic_map_identity()
        _ = self.forward( tf.zeros([1, self.dim_flow]) )
        #[self.LAYERS[i].forward(tf.zeros([1, self.dim_flow])) for i in  self.inds_layers_forward]
        self.n_trainable_tensors = len(self.trainable_weights)

        if IC_map is not None: self.IC_map = IC_map
        else: pass
        if IC_map is not None: self.forward_dims = [None,self.IC_map.n_atoms,3]
        else: self.forward_dims = [None,self.dim_flow]
        self.inverse_dims = [None,self.dim_flow]

        #self.forward = tf.function(self.forward, input_signature = [tf.TensorSpec(shape=self.forward_dims, dtype=DTYPE_tf)])
        #self.inverse = tf.function(self.inverse, input_signature = [tf.TensorSpec(shape=self.inverse_dims, dtype=DTYPE_tf)])

        #################

        ## generic should be inherited (TODO):
        self.log_prior_uniform = - self.dim_flow*np.log(2.0)
        self.evaluate_log_prior_ = lambda z : self.log_prior_uniform
        self.sample_prior_ = lambda batch_size : tf.random.uniform(shape=[batch_size, self.dim_flow], minval=-1.0,  maxval=1.0)
        self.sample_base_ = self.sample_prior_ 
        self.ln_base_ =  self.evaluate_log_prior_
        ##

        if verbose: self.print_model_size()
        else: pass

        self.store_initial_parameters_()

        self.reset_optimiser(optimiser_LR_decay)
        self.training_batch_size = None

        self.forward_graph_ = tf.function(self.forward_graph_)
        self.inverse_graph_ = tf.function(self.inverse_graph_)
        #self.ln_model_for_step_ML_ = tf.function(self.ln_model_for_step_ML_)
        self.step_ML_graph_ = tf.function(self.step_ML_graph_)

    def reset_optimiser(self, optimiser_LR_decay):
        self.learning_rate, self.rate_decay = optimiser_LR_decay
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate,
                                                  decay = self.rate_decay)

    def forward_(self, x):
        ladJ_forward = 0.0
        x, ladj = self.IC_map.forward(x) ; ladJ_forward += ladj
        for i in self.inds_layers_forward:
            x, ladJ = self.LAYERS[i].forward(x) ; ladJ_forward += ladJ
        return x, ladJ_forward 

    def inverse_(self, x):
        ladJ_inverse = 0.0
        for i in self.inds_layers_inverse:
            x, ladJ = self.LAYERS[i].inverse(x) ; ladJ_inverse += ladJ
        x, ladj = self.IC_map.inverse(x) ; ladJ_inverse += ladj
        return x, ladJ_inverse

    #tf.function
    def forward_graph_(self, x):
        return self.forward_(x)
    #tf.function
    def inverse_graph_(self, x):
        return self.inverse_(x)

    def forward(self, xyz, batch_size=None):
        xyz = np2tf_(xyz)
        if batch_size is None:
            return self.forward_graph_(xyz)
        else:
            return batched_evaluation_(_function = self.forward_graph_,
                                       _input = xyz,
                                       batch_size = batch_size,
                                       function_output_dims = [[self.dim_flow],[1]])
    def inverse(self, z, batch_size=None):
        z = np2tf_(z)
        if batch_size is None:
            return self.inverse_graph_(z)
        else:
            return batched_evaluation_(_function = self.inverse_graph_,
                                       _input = z,
                                       batch_size = batch_size,
                                       function_output_dims = [self.forward_dims[1:],[1]])

    def sample_model(self, m, detach_outputs=True, batch_size=None):
        z = self.sample_base_(m)
        x, ladJ = self.inverse(z, batch_size=batch_size)
        outputs = [x, self.ln_base_(z) - ladJ]
        if detach_outputs: return [np.array(_) for _ in outputs]
        else: return outputs
        
    def ln_model(self, x, detach_outputs=True, batch_size=None):
        x = np2tf_(x)
        if self.training_batch_size is not None:
            if self.training_batch_size == batch_size:
                outputs = self.ln_model_for_step_ML_(x)
            else: 
                z, ladJ = self.forward(x, batch_size=batch_size)
                outputs = [z, self.ln_base_(z) + ladJ]
        else:
            z, ladJ = self.forward(x, batch_size=batch_size)
            outputs = [z, self.ln_base_(z) + ladJ]
            
        if detach_outputs: return [np.array(_) for _ in outputs]
        else: return outputs

    ##

    #tf.function
    def ln_model_for_step_ML_(self,x):
        z, ladJ = self.forward_(x)
        return z, self.ln_base_(z) + ladJ

    #tf.function
    def step_ML_graph_(self, xyz_batch):
        # r_batch : (m,n_atoms,3)
        with tf.GradientTape() as tape:
            ln_p = self.ln_model_for_step_ML_(xyz_batch)[-1]
            loss = - tf.reduce_mean(ln_p)
        grads = tape.gradient(loss, self.trainable_variables)
        grads = [x for x in grads if x is not None] # remove this line when that thing is fixed.
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def step_ML_(self, xyz, u=None, batch_size=1000):
        # data : (m,n_atoms,3) : MD data
        self.training_batch_size = batch_size
        inds_rand = np.random.choice(xyz.shape[0], batch_size, replace=False)
        xyz_batch = np2tf_(xyz[inds_rand])
        AVMD_estimate_of_batch_entropy = self.step_ML_graph_(xyz_batch).numpy()
        if u is None: return AVMD_estimate_of_batch_entropy, AVMD_estimate_of_batch_entropy
        else: 
            AVMD_estimate_of_batch_Helmholtz_conf_fe = u[inds_rand].mean() - AVMD_estimate_of_batch_entropy
            return AVMD_estimate_of_batch_Helmholtz_conf_fe, AVMD_estimate_of_batch_entropy

    ###################################################################################

    ## generic functions: [TODO: should be inherited]

    def print_model_size(self):
        ws = self.trainable_weights
        n_trainable_variables = sum([np.product(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',n_trainable_variables,'trainable parameters in this model, among', len(ws),'trainable_variables.' )
        shapes = [tuple(x.shape) for x in ws]
        shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]
        self.shapes_trainable_variables = [''.join([(' ' * (8 - len(y))) + y for y in [x.split(' ')  for x in shapes_str][i]]) for i in range(len(shapes)//2)]
        print('[NB: To see dimensionalities of the trainable variables print(list(self.shapes_trainable_variables)).] ')

    def save_model(self, path_and_name : str):
        save_pickle_([self.init_args, self.trainable_variables], path_and_name)
        
    @staticmethod
    def load_model(path_and_name : str):
        init_args, ws = load_pickle_(path_and_name)

        # To fix a bug when loading a bit older files, also IC_MAP.py cannot be renamed when loading them.
        if len(init_args) == 14:
            init_args = init_args[:3] + [[0.001,0.0001]] + init_args[3:]
            print('loading older model, optimiser_LR_decay set to [0.001,0.0001] to prevent error. If training further can adjust this via self.reset_optimiser(optimiser_LR_decay).')
        elif len(init_args) not in [14,15]:
            print('the init_args were changed again, may have segmentation error when loading, unless adjusted for here')
        else: pass

        loaded_model = (lambda f, args : f(*args))(MODEL_3, init_args)
        for i in range(len(ws)):
            loaded_model.trainable_variables[i].assign(ws[i])
        return loaded_model

    def store_initial_parameters_(self):
        self.initial_parameters = []
        for i in range(self.n_trainable_tensors):
            self.initial_parameters.append(tf.Variable(self.trainable_variables[i]))

    def replace_paremeters(self, list_params):
        for i in range(self.n_trainable_tensors):
            self.trainable_variables[i].assign(list_params[i])
        self.store_initial_parameters_()

    def forward_np(self, x):
        x, ladJ = self.forward( tf.constant(x, dtype=tf.float32) )
        return x.numpy(), ladJ.numpy()
        
    def inverse_np(self, x):
        x, ladJ  = self.inverse( tf.constant(x, dtype=tf.float32) ) 
        return x.numpy(), ladJ.numpy()

    def sample(self, n_samples):
        return self.inverse_np(self.sample_prior_(n_samples))

