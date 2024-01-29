import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
RQS_class = tfp.bijectors.RationalQuadraticSpline

##

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                 dims_outputs : list,
                 outputs_activations : list = None,
                 dims_hidden : list = [100],
                 hidden_activation = tf.nn.silu,
                 **kwargs):
        super().__init__(**kwargs)

        ''' MLP : class : Multilayer Perceptron.

        Inputs:
            dims_outputs : list of ints.
            outputs_activations : list of functions, or None (output layer(s) linear).
            dims_hidden : list of ints. Each int indicates how many hidden units. The length of this list determines how many hidden layers.
            hidden_activation : nonlinear function. After each hidden later this nonlinearity is applied.

        '''

        n_hidden_layers = len(dims_hidden)
        n_output_layers = len(dims_outputs)

        if outputs_activations is None: outputs_activations = ['linear']*n_output_layers
        else: pass

        self.hidden_layers = [tf.keras.layers.Dense(dims_hidden[i], activation = hidden_activation) for i in range(n_hidden_layers)]
        self.output_layers = [tf.keras.layers.Dense(dims_outputs[j], activation = outputs_activations[j]) for j  in range(n_output_layers)]

    def call(self, x, drop_rate = 0.0):
        '''
        Inputs:
            x : (m,d) shaped tensor of data. m = batch size. d = number of variables.
            drop_rate : float in range [0,1]. 
                Default is 0.0, and always set to zero when not training.
                During training around 0.1 is approximate heuristic. [Not used.]
        Output:
            ys : list of tensors with shapes (m,dims_outputs[i]) for every output layer i.
        '''
        for layer in self.hidden_layers:
            x = layer(x)
        if drop_rate > 0.0: x = tf.keras.layers.Dropout(rate = drop_rate)(x, training=True)
        else: pass
        ys = [layer(x) for layer in self.output_layers]
        return ys

##

def bin_positons_(MLP_output, # (m, dim*n_bins)
                  dim,
                  n_bins,
                  domain_width,
                  min_bin_width,
                 ):
    MLP_output = tf.reshape(MLP_output, [-1, dim, n_bins])
    c = domain_width - n_bins*min_bin_width
    bin_positons = tf.nn.softmax(MLP_output, axis=-1) * c + min_bin_width
    return bin_positons # (m, dim, n_bins)

def knot_slopes_(MLP_output, # (m, dim*(n_bins-1))
                 dim,
                 n_bins,
                 min_knot_slope,
                ):
    MLP_output = tf.reshape(MLP_output, [-1, dim, n_bins - 1])
    knot_slopes = tf.nn.softplus(MLP_output) + min_knot_slope
    return knot_slopes # (m, dim, n_bins-1)

def rqs_(x,         # (m, dim) shaped tensor of data. m = batch size. dim = number of variables. # all values in range xy_range
         w,         # arbitrary unconstrained MLP_output with shape (m, dim*n_bins)
         h,         # arbitrary unconstrained MLP_output with shape (m, dim*n_bins)
         s,         # arbitrary unconstrained MLP_output with shape (m, dim*(n_bins-1))
         forward = True,
         xy_range = [0.0, 1.0],
         min_bin_width = 0.001,
         min_knot_slope = 0.001,
        ):
    ''' rqs : rational quadratic spline function transforming x (forward or not forward), parametrised by unconstrained MLP outputs [w,h,s]
        Outputs: 
            y : (m, dim) shaped tensor of elementwise transformed data. # all values in range xy_range
            ladJ : (m, dim) shaped tensor of elementwise log_det_jacobians (=ln(dy[i]/dx[i]) i=0,...,dim-1 in this case)
    '''
    m, dim = x.shape
    n_bins = w.shape[1] // dim
    
    xy_min, xy_max = xy_range
    domain_width = xy_max - xy_min

    bin_positons_x = bin_positons_(w,
                                   dim = dim,
                                   n_bins = n_bins,
                                   domain_width = domain_width,
                                   min_bin_width = min_bin_width) # (m, dim, n_bins)
    bin_positons_y = bin_positons_(h,
                                   dim = dim,
                                   n_bins = n_bins,
                                   domain_width = domain_width,
                                   min_bin_width = min_bin_width) # (m, dim, n_bins)
    
    knot_slopes = knot_slopes_(s, dim = dim, n_bins = n_bins, min_knot_slope = min_knot_slope) # (m, dim, n_bins-1)
 
    RQS_obj = RQS_class(bin_widths = bin_positons_x,
                        bin_heights = bin_positons_y,
                        knot_slopes = knot_slopes,
                        range_min = xy_min,
                       )
    if forward:
        y = RQS_obj.forward(x)
        ladJ = RQS_obj.forward_log_det_jacobian(x)
    else:
        y = RQS_obj.inverse(x)
        ladJ = RQS_obj.inverse_log_det_jacobian(x)
    return y, ladJ

def shift_(x, shifts, forward=True, xy_range = [0.0,1.0]):
    A, B = xy_range
    if forward: return tf.math.floormod(x+shifts - A, B-A) + A
    else: return       tf.math.floormod(x-shifts - A, B-A) + A
    
def rqs_with_periodic_shift_(x,           # (m, dim) shaped tensor of periodic data. m = batch size. dim = number of variables. # all values in periodic interval xy_range
                             list_w,      # list of k arbitrary unconstrained MLP_outputs with shape (m, dim*n_bins) 
                             list_h,      # list of k arbitrary unconstrained MLP_outputs with shape (m, dim*n_bins) 
                             list_shifts, # list of k arbitrary unconstrained MLP_outputs with shape (m, dim) (or scalar constants)
                             list_s,      # list of k arbitrary unconstrained MLP_output with shape (m, dim*(n_bins-1))
                             forward = True,
                             xy_range = [0.0, 1.0],
                             min_bin_width = 0.001,
                             min_knot_slope = 0.001,
                            ):
    ''' same as rqs but with periodic shifts which may or may not be trainable
            Input parameters are lists of k unconstrained MLP_outputs, where k = 2.
                Each variable transformed twice, once with shift being applied and once without.
                This ensures representational power of rqs is extended to the entire domain (xy_range).
    '''
    n_transforms = len(list_h)

    ladJsum = 0.0

    if forward: inds_list = [i for i in range(n_transforms)]
    else:       inds_list = [n_transforms-1-i for i in range(n_transforms)]
    
    for i in inds_list:
        x = shift_(x, list_shifts[i], forward=True, xy_range = xy_range)
        x, ladJ = rqs_(x,
                       w = list_w[i],
                       h = list_h[i],
                       s = list_s[i],
                       forward = forward,
                       xy_range = xy_range,
                       min_bin_width = min_bin_width,
                       min_knot_slope = min_knot_slope,
                      ) ; ladJsum += ladJ
        x = shift_(x, list_shifts[i], forward=False, xy_range = xy_range)
    return x, ladJsum # (m, dim), (m, dim)

##

def sum_(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)

PI = 3.1415926535897932384626433832795028841971693993751058209

def cos_sin_(x, nk:int=1):
    x*=PI
    output = []
    for k in range(1,nk+1):
        output.append(tf.cos(k*x))
        output.append(tf.sin(k*x))
    return tf.concat(output, axis=-1)

def cos_sin_1_(x):
    x*=PI
    return tf.concat( [tf.cos(x),tf.sin(x)], axis=1 )

def broadcasting_app_axis1_(x, n):
    # want to reshape x with shape (m,n*d) into y with shape (n,m,d), keeping axis m fixed.
    m = x.shape[1] // n
    inds_axis_1 = [tf.range(m)+i*m for i in range(n)]
    y = tf.stack([tf.gather(x,inds_axis_1[i], axis=1) for i in range(n)])
    return y # with shape (n,m,d)

class SPLINE_COUPLING_LAYER(tf.keras.layers.Layer):
    def __init__(self,
                 periodic_mask, # (dim_flow,) shaped list ; 1 if marginal variable periodic, 0 if marginal variable non-periodic
                 cond_mask, # (dim_flow,) shaped list ; 1 if variable is being transformed, 0 if variable is being used to condition this transformation

                 n_bins_periodic = 8, # resolution of spline functions involved with periodic variables
                 number_of_splines_periodic = 2,
                 n_bins_other = 8, # resolution of spline functions involved with nonperiodic variables
                 
                 n_hidden = 1, # hyperparameter overridden when dims_hidden is not None
                 hidden_activation = tf.nn.silu,

                 flow_range = [-1.0,1.0], # hyperparameter not adjustable here!
                 min_bin_width = 0.001,
                
                 trainable_slopes = True, # hyperparameter not adjustable here!
                 min_knot_slope = 0.001,
                 
                 dims_hidden = None, # if None, dims_hidden set to [output dim]

                 nk_for_periodic_MLP_encoding = 1, # dimensionality of sine cosine encoding of periodic variables before input to MLS.
                ):
        super().__init__()
        periodic_mask = np.array(periodic_mask).flatten()
        cond_mask = np.array(cond_mask).flatten()
        self.periodic_mask = periodic_mask
        self.cond_mask = cond_mask

        self.n_bins_P = n_bins_periodic
        self.n_splines_P = number_of_splines_periodic 
        self.n_bins_O = n_bins_other

        self.n_hidden = n_hidden 
        self.hidden_activation = hidden_activation 
        self.joined_MLPs = True #
        self.flow_range = [-1.0,1.0]
        self.min_bin_width = min_bin_width

        self.trainable_slopes = True
        self.min_knot_slope = min_knot_slope

        self.dims_hidden = dims_hidden

        self.nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding
        ##

        self.n_variables = len(periodic_mask)
        if len(cond_mask) != self.n_variables: print('!! SPLINE_COUPLING_LAYER : lengths of both masks should be equal')
        else: pass

        self.inds_A_P = np.where((cond_mask==1)&(periodic_mask==1))[0]  # inds periodic in 1st part
        self.inds_A_O = np.where((cond_mask==1)&(periodic_mask!=1))[0]  # inds other in 1st part
        self.inds_A_cP = np.where((cond_mask==0)&(periodic_mask==1))[0] # inds not flowing periodic in 1st part
        self.inds_A_cO = np.where((cond_mask==0)&(periodic_mask!=1))[0] # inds not flowing other in 1st part

        cat_inds_A = np.concatenate([self.inds_A_P, self.inds_A_O, self.inds_A_cP, self.inds_A_cO])
        self.inds_unpermute_A = np.array([int(np.where(cat_inds_A == i)[0]) for i in range(len(cat_inds_A))])

        self.inds_B_P = np.where((cond_mask==0)&(periodic_mask==1))[0]  # inds periodic in 2nd part
        self.inds_B_O = np.where((cond_mask==0)&(periodic_mask!=1))[0]  # inds other in 2nd part
        self.inds_B_cP = np.where((cond_mask==1)&(periodic_mask==1))[0] # inds not flowing periodic in 2nd part
        self.inds_B_cO = np.where((cond_mask==1)&(periodic_mask!=1))[0] # inds not flowing other in 2st part

        cat_inds_B = np.concatenate([self.inds_B_P, self.inds_B_O, self.inds_B_cP, self.inds_B_cO])
        self.inds_unpermute_B = np.array([int(np.where(cat_inds_B == i)[0]) for i in range(len(cat_inds_B))])

        self.n_A_P = len(self.inds_A_P)
        self.n_A_O = len(self.inds_A_O)
        self.n_A_cP = len(self.inds_A_cP)
        self.n_A_cO = len(self.inds_A_cO)

        self.n_B_P = len(self.inds_B_P)
        self.n_B_O = len(self.inds_B_O)
        self.n_B_cP = len(self.inds_B_cP)
        self.n_B_cO = len(self.inds_B_cO)
        ##

        if self.trainable_slopes:
            dim_slopes_A_P = self.n_A_P*(self.n_bins_P-1) ; dim_slopes_B_P = self.n_B_P*(self.n_bins_P-1)
            dim_slopes_A_O = self.n_A_O*(self.n_bins_O-1) ; dim_slopes_B_O = self.n_B_O*(self.n_bins_O-1)
        else: # not
            dim_slopes_A_P = dim_slopes_B_P = 0
            dim_slopes_A_O = dim_slopes_B_O = 0 

        ''' 
        This is a coupling layer: All variables enter into this layer and get split into sets A and B, 
        based on conditioning mask (cond_mask). Transformation of variables in set A is conditioned on 
        variables in set B, carried out by self.A_ function. Then, variables of set B are transformed 
        conditional on outputs of self.A_ (still in set A) by self.B_ function. In both sets A and B 
        there can be a mixture of periodic or nonperiodic variables. This involves variables being 
        transformed (by spline [as a function of MLP outputs]), and those involved in conditioning 
        (inputs to MLPs). In both cases periodic variables must be treated differently because they are 
        non-Euclidean. Tensorflow does not support item assignment, thus any simple indexing exercise 
        is not compact. By the end of coupling layer all variables return to their original permutation.
        '''

        self.output_dims_MLP_A = []
        if self.n_A_P > 0:

            self.output_dims_MLP_A.append( self.n_splines_P * (self.n_A_P*self.n_bins_P*2 + dim_slopes_A_P + self.n_A_P) ) 
            if self.n_A_O > 0:
                self.output_dims_MLP_A.append( self.n_A_O*self.n_bins_O*2 + dim_slopes_A_O )
                self.A_ = self.A_PO_
            else: 
                self.A_ = self.A_P_
        else:
            self.output_dims_MLP_A.append( self.n_A_O*self.n_bins_O*2 + dim_slopes_A_O )
            self.A_ = self.A_O_
            
        if dims_hidden is None: dims_hidden_A = [sum(self.output_dims_MLP_A)]*n_hidden
        else: dims_hidden_A = dims_hidden
        self.MLP_A = MLP(dims_outputs = self.output_dims_MLP_A,
                         outputs_activations = None,
                         dims_hidden = dims_hidden_A,
                         hidden_activation = hidden_activation)

        self.output_dims_MLP_B = []
        if self.n_B_P > 0:
            self.output_dims_MLP_B.append( self.n_splines_P * (self.n_B_P*self.n_bins_P*2 + dim_slopes_B_P + self.n_B_P) ) 
            if self.n_B_O > 0:
                self.output_dims_MLP_B.append( self.n_B_O*self.n_bins_O*2 + dim_slopes_B_O )
                self.B_ = self.B_PO_
            else: 
                self.B_ = self.B_P_
        else:
            self.output_dims_MLP_B.append( self.n_B_O*self.n_bins_O*2 + dim_slopes_B_O )
            self.B_ = self.B_O_
            
        if dims_hidden is None: dims_hidden_B = [sum(self.output_dims_MLP_B)]*n_hidden
        else: dims_hidden_B = dims_hidden
        self.MLP_B = MLP(dims_outputs = self.output_dims_MLP_B,
                         outputs_activations = None,
                         dims_hidden = dims_hidden_B,
                         hidden_activation = hidden_activation)
        ##

        if nk_for_periodic_MLP_encoding == 1: self.cos_sin_ = cos_sin_1_ # may be faster?
        else: self.cos_sin_ = lambda x : cos_sin_(x, nk=nk_for_periodic_MLP_encoding)

    def A_PO_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xAP = tf.gather(x, self.inds_A_P, axis=1)
        xAO = tf.gather(x, self.inds_A_O, axis=1)

        xAcP = tf.gather(x, self.inds_A_cP, axis=1)
        xAcO = tf.gather(x, self.inds_A_cO, axis=1)
        xAc = tf.concat([self.cos_sin_(xAcP), xAcO], axis=1)
        pAP, pAO = self.MLP_A(xAc)

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pAP = broadcasting_app_axis1_(pAP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_A_P*self.n_bins_P
        yAP,ladJ = rqs_with_periodic_shift_(xAP,                                        # (m,dim)
                                            list_w = pAP[:,:,:n],                       # (m,dim*n_bins) * n_transforms
                                            list_h = pAP[:,:,n:2*n],                    # (m,dim*n_bins) * n_transforms
                                            list_shifts = pAP[:,:,2*n:2*n+self.n_A_P],  # (m,dim)
                                            list_s = pAP[:,:,2*n+self.n_A_P:],
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        m = self.n_A_O*self.n_bins_O
        yAO,ladJ = rqs_(xAO,
                        w = pAO[:,:m],
                        h = pAO[:,m:2*m], 
                        s = pAO[:,2*m:], 
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order (join):

        cat_y = tf.concat([yAP, yAO, xAcP, xAcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_A, axis=1)

        return y, ladJ_sum

    def B_PO_(self, x, forward = True):
        """ roles of flowing vs. conditinoing swapped
        """
        # split all types of variables and run transformations:
        
        xBP = tf.gather(x, self.inds_B_P, axis=1)
        xBO = tf.gather(x, self.inds_B_O, axis=1)

        xBcP = tf.gather(x, self.inds_B_cP, axis=1)
        xBcO = tf.gather(x, self.inds_B_cO, axis=1)
        xBc = tf.concat([self.cos_sin_(xBcP), xBcO], axis=1)
        pBP, pBO = self.MLP_B(xBc)

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pBP = broadcasting_app_axis1_(pBP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_B_P*self.n_bins_P
        yBP,ladJ = rqs_with_periodic_shift_(xBP,                                        # (m,dim)
                                            list_w = pBP[:,:,:n],                       # (m,dim*n_bins) * n_transforms
                                            list_h = pBP[:,:,n:2*n],                    # (m,dim*n_bins) * n_transforms
                                            list_shifts = pBP[:,:,2*n:2*n+self.n_B_P],  # (m,dim)
                                            list_s = pBP[:,:,2*n+self.n_B_P:],
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        m = self.n_B_O*self.n_bins_O
        yBO,ladJ = rqs_(xBO,
                        w = pBO[:,:m],
                        h = pBO[:,m:2*m], 
                        s = pBO[:,2*m:], 
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order (join):

        cat_y = tf.concat([yBP, yBO, xBcP, xBcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_B, axis=1)

        return y, ladJ_sum

    def A_P_(self, x, forward = True):
        
        # split all types of variables and run transformations:
        
        xAP = tf.gather(x, self.inds_A_P, axis=1)

        xAcP = tf.gather(x, self.inds_A_cP, axis=1)
        xAcO = tf.gather(x, self.inds_A_cO, axis=1)
        xAc = tf.concat([self.cos_sin_(xAcP), xAcO], axis=1)
        pAP = self.MLP_A(xAc)[0] # raw params.

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pAP = broadcasting_app_axis1_(pAP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_A_P*self.n_bins_P
        yAP,ladJ = rqs_with_periodic_shift_(xAP,                          # (m,dim)
                                            list_w = pAP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                            list_h = pAP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                            list_shifts = pAP[:,:,2*n:2*n+self.n_A_P],  # (m,dim)
                                            list_s = pAP[:,:,2*n+self.n_A_P:],
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        # put everything back in the right order (join):

        cat_y = tf.concat([yAP, xAcP, xAcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_A, axis=1)

        return y, ladJ_sum

    def B_P_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xBP = tf.gather(x, self.inds_B_P, axis=1)

        xBcP = tf.gather(x, self.inds_B_cP, axis=1)
        xBcO = tf.gather(x, self.inds_B_cO, axis=1)
        xBc = tf.concat([self.cos_sin_(xBcP), xBcO], axis=1)
        pBP = self.MLP_B(xBc)[0] # raw params.

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pBP = broadcasting_app_axis1_(pBP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_B_P*self.n_bins_P
        yBP,ladJ = rqs_with_periodic_shift_(xBP,                          # (m,dim)
                                            list_w = pBP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                            list_h = pBP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                            list_shifts = pBP[:,:,2*n:2*n+self.n_B_P],  # (m,dim)
                                            list_s = pBP[:,:,2*n+self.n_B_P:],
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        # put everything back in the right order (join):

        cat_y = tf.concat([yBP, xBcP, xBcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_B, axis=1)

        return y, ladJ_sum

    def A_O_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xAO = tf.gather(x, self.inds_A_O, axis=1)

        xAcP = tf.gather(x, self.inds_A_cP, axis=1)
        xAcO = tf.gather(x, self.inds_A_cO, axis=1)
        xAc = tf.concat([self.cos_sin_(xAcP), xAcO], axis=1)
        pAO = self.MLP_A(xAc)[0] # raw params.

        ladJ_sum = 0.0

        m = self.n_A_O*self.n_bins_O
        yAO,ladJ = rqs_(xAO,
                        w = pAO[:,:m],
                        h = pAO[:,m:2*m], 
                        s = pAO[:,2*m:],
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order (join):

        cat_y = tf.concat([yAO, xAcP, xAcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_A, axis=1)

        return y, ladJ_sum

    def B_O_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xBO = tf.gather(x, self.inds_B_O, axis=1)

        xBcP = tf.gather(x, self.inds_B_cP, axis=1)
        xBcO = tf.gather(x, self.inds_B_cO, axis=1)
        xBc = tf.concat([self.cos_sin_(xBcP), xBcO], axis=1)
        pBO = self.MLP_B(xBc)[0] # raw params.

        ladJ_sum = 0.0

        m = self.n_B_O*self.n_bins_O
        yBO,ladJ = rqs_(xBO,
                        w = pBO[:,:m],
                        h = pBO[:,m:2*m], 
                        s = pBO[:,2*m:], 
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order (join):

        cat_y = tf.concat([yBO, xBcP, xBcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_B, axis=1)

        return y, ladJ_sum

    def forward(self,x):
        ladJ_forward = 0.0
        x, ladJ = self.A_(x, forward=True) ; ladJ_forward += ladJ
        x, ladJ = self.B_(x, forward=True) ; ladJ_forward += ladJ
        return x, ladJ_forward

    def inverse(self,x):
        ladJ_inverse = 0.0 
        x, ladJ = self.B_(x, forward=False) ; ladJ_inverse += ladJ
        x, ladJ = self.A_(x, forward=False) ; ladJ_inverse += ladJ
        return x, ladJ_inverse
