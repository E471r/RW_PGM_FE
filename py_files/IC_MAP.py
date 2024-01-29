from rdkit import Chem

import numpy as np

import tensorflow as tf ; DTYPE_tf = tf.float32 ; clamp_float_tf_ = lambda x : tf.cast(x, dtype=DTYPE_tf)

import tensorflow_probability as tfp

PI = 3.14159265358979323846264338327950288

##

_clip_low_at_ = 1e-6
_clip_high_at_ = 1e+6

clamp_positive_ = lambda x : tf.clip_by_value(x, _clip_low_at_, _clip_high_at_) 

def get_distance_tf_(R, inds_2_atoms):
    # R            : (..., # atoms, 3)
    # inds_2_atoms : (2,)
    A,B = inds_2_atoms
    rA = R[...,A,:]  # (...,3)
    rB = R[...,B,:]  # (...,3)
    vBA = rA - rB    # (...,3)
    d = clamp_positive_(tf.norm(vBA, axis=-1, keepdims=True)) # (...,1)
    dd_drA = vBA / d # (...,3)
    return d, dd_drA # (...,1), (...,3)

def get_angle_tf_(R, inds_3_atoms):
    # R            : (..., # atoms, 3)
    # inds_3_atoms : (3,)

    A,B,C = inds_3_atoms
    rA = R[:,A,:] # (...,3)
    rB = R[:,B,:] # (...,3)
    rC = R[:,C,:] # (...,3)

    vBA = rA - rB # (...,3)
    vBC = rC - rB # (...,3)

    NORM_vBA = clamp_positive_(tf.norm(vBA, axis=-1, keepdims=True)) # (...,1)
    NORM_vBC = clamp_positive_(tf.norm(vBC, axis=-1, keepdims=True)) # (...,1)

    uBA = vBA / NORM_vBA # (...,3)
    uBC = vBC / NORM_vBC # (...,3)

    dot = tf.reduce_sum(uBA*uBC, axis=-1, keepdims=True)             # (...,1)
    dot = tf.clip_by_value(dot, -1.0, 1.0)                           # (...,1)
    
    theta = tf.acos(dot) # (...,1)
    theta = tf.clip_by_value(theta, _clip_low_at_, PI-_clip_low_at_) # (...,1)

    one_minus_x_sq = 1.0 - dot**2                                    # (...,1)
    one_minus_x_sq = tf.clip_by_value(one_minus_x_sq, _clip_low_at_, 1.0)
    dacos_dx = - 1.0 / tf.sqrt(one_minus_x_sq)                       # (...,1)
    dtheta_drA = dacos_dx * (uBC - uBA * dot) / NORM_vBA

    return theta, dtheta_drA                                         # (...,1), (...,3)

def get_torsion_tf_(R, inds_4_atoms):
    # Reference: https://github.com/noegroup/bgflow
    # R            : (..., # atoms, 3)
    # inds_4_atoms : (4,)
    
    A,B,C,D = inds_4_atoms
    rA = R[...,A,:] # (...,3)
    rB = R[...,B,:] # (...,3)
    rC = R[...,C,:] # (...,3)
    rD = R[...,D,:] # (...,3)
    
    vBA = rA - rB   # (...,3)
    vBC = rC - rB   # (...,3)
    vCD = rD - rC   # (...,3)
    
    NORM_vBC = clamp_positive_(tf.norm(vBC, axis=-1, keepdims=True)) # (...,1)
    uBC = vBC / NORM_vBC # (m,3)
    
    w = vCD - tf.reduce_sum(vCD*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    v = vBA - tf.reduce_sum(vBA*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    
    uBC1 = uBC[:,0] # (...,)
    uBC2 = uBC[:,1] # (...,)
    uBC3 = uBC[:,2] # (...,)
    
    zero = tf.zeros_like(uBC1) # (...,)
    S = tf.stack([tf.stack([ zero, uBC3,-uBC2],axis=-1),
                  tf.stack([-uBC3, zero, uBC1],axis=-1),
                  tf.stack([ uBC2,-uBC1, zero],axis=-1)],axis=-1) # (...,3,3)
    
    y = tf.expand_dims(tf.einsum('...j,...jk,...k->...',w,S,v), axis=-1) # (...,1)
    x = tf.expand_dims(tf.einsum('...j,...j->...',w,v), axis=-1)         # (...,1)
    
    phi = tf.math.atan2(y,x) # (...,1)
    
    denominator = clamp_positive_(x**2 + y**2) # (...,1)
    S_transpose_w = tf.einsum('...jk,...j->...k',S,w) # (...,3)
    ##
    # numerator = x*S_transpose_w - y*w # (...,3)
    safe = tf.eye(3)[None, :, :] - uBC[..., None] * uBC[..., None, :] # (m,3,3)  # in case second derivative.
    numerator  = x*tf.einsum('ij,ijk->ik', S_transpose_w, safe) - y*tf.einsum('ij,ijk->ik', w, safe) # (m,3)
    ##
    dphi_drA = numerator/denominator # (...,3)
    
    return phi, dphi_drA # (...,1), (...,3)

def det_3x3_(M, keepdims=False):
    # M : (...,3,3)
    return tf.reduce_sum( tf.linalg.cross(M[...,0], M[...,1]) * M[...,2], axis=-1, keepdims=keepdims)

def NeRF_tf_(d, theta, phi, rB, rC, rD, dont_return_ladJ = False):

    # xA -> rA : Reference: DOI 10.1002/jcc.20237
    # xA = [d,theta,phi] ; d : (...,) ; theta : (...,) ; phi : (...,)
    # constants: rB : (...,3) ; rD : (...,3) ; rD : (...,3)

    vCB = rB-rC # (...,3)
    vDC = rC-rD # (...,3)

    NORM_vCB = tf.norm(vCB ,axis=-1, keepdims=True) # (...,1)
    #NORM_vCB = clamp_positive_(NORM_vCB) # (...,1)
    NORM_vDC = tf.norm(vDC, axis=-1, keepdims=True) # (...,1)
    #NORM_vDC = clamp_positive_(NORM_vDC) # (...,1)
    
    uCB = vCB / NORM_vCB # (...,3)
    uDC = vDC / NORM_vDC # (...,3)
    
    nv = tf.linalg.cross(uDC,uCB)                 # (...,3)
    NORM_nv = tf.norm(nv, axis=-1, keepdims=True) # (...,1)
    #NORM_nv = clamp_positive_(NORM_nv) # (...,1)
    nv /= NORM_nv                       # (...,3)
    
    M = tf.stack([ uCB, tf.linalg.cross(nv,uCB), nv ], axis=-1) # (...,3,3)

    the = PI - theta # (m,)
    c_t = tf.cos(the) ; s_t = tf.sin(the) # (...,)
    c_p = tf.cos(phi) ; s_p = tf.sin(phi) # (...,)

    v = tf.stack([ d*c_t, d*s_t*c_p, d*s_t*s_p ], axis=-1) # (...,3)

    rA = rB + tf.einsum('...ij,...j->...i', M, v) # (....,3)

    if dont_return_ladJ:
        return rA, 0.0 # (...,3), (,)
    else:
        zero = tf.zeros_like(c_t) # (...,)
        partials = tf.stack([tf.stack([  c_t,     -d*s_t,      zero      ], axis=-1), # (...,3)
                             tf.stack([  s_t*c_p,  d*c_t*c_p, -d*s_t*s_p ], axis=-1), # (...,3)
                             tf.stack([  s_t*s_p,  d*c_t*s_p,  d*s_t*c_p ], axis=-1), # (...,3)
                            ], axis=-2) # (...,3,3)
        jacobian_drA_dxA = tf.einsum('...ij,...jk->...ik', M, partials) # (...,3,3)
        #return rA, jacobian_drA_dxA # (...,3), (...,3,3)

        ladJ = tf.math.log(tf.abs(det_3x3_(jacobian_drA_dxA, keepdims=True)))
        return rA, ladJ # (...,3), (...,1)

'''
def ladJ_from_list_of_jacobians_tf_(jacobians):
    # m = number of frames
    # jacobians : list ~ [(m,3,3),(m,3,3),...] of length n_IC
    Js = tf.stack(jacobians, axis=1) # (m, n_IC, 3, 3)
    detJs = det_3x3_(Js, keepdims=False) # (m, n_IC)
    log_abs_detJs = tf.math.log( tf.abs( detJs ) ) # (m, n_IC)
    ladJ = tf.reduce_sum(log_abs_detJs, axis=1, keepdims=True) # (m,1)
    return ladJ # (m,1)
'''

def r_to_x_(R, inds_abcd, concise_ladJ = True):
    # R : (m,#atoms,3) ; m = batch size.
    b, db_drA = get_distance_tf_( R, inds_abcd[:2] ) # (m,1), (m,3)
    a, da_drA = get_angle_tf_(    R, inds_abcd[:3] ) # (m,1), (m,3)
    t, dt_drA = get_torsion_tf_(  R, inds_abcd[:4] ) # (m,1), (m,3)
    xA = tf.concat([b,a,t],axis=-1) # (m,3)

    if concise_ladJ:
         ladJ = - tf.math.log(tf.sin(a)*(b**2)) # (m,1)
         return xA, ladJ # (m,3), (m,1)
    else:
        jacobian_dxA_drA = tf.stack([db_drA, da_drA, dt_drA], axis=-1) # (m,3,3)
        ladJ = tf.math.log(tf.abs(det_3x3_(jacobian_dxA_drA, keepdims=True))) # (m,1)
        return xA, ladJ # (m,3), (m,1)

def R_to_X_(R, inds_ABCD, concise_ladJ = True):
    # R : (m,n,3) ; m = batch size, n = #atoms
    n = R.shape[1]
    X = [] ; ladJ_RX = 0.0
    for i in range(n):
        x, ladJ = r_to_x_(R, inds_ABCD[i], concise_ladJ = concise_ladJ) # (m,3), (m,1)
        X.append(x) ; ladJ_RX += ladJ
    X = tf.stack(X, axis=1) # (m,n,3)
    return X, ladJ_RX       # (m,n,3), (m,1)

def R_to_X_mixed_(R, inds_ABCD, inds_IC, concise_ladJ = True):
    n = R.shape[1]
    X_mixed = [] ; ladJ_RX = 0.0
    for i in range(n):
        if i in inds_IC:
            x, ladJ = r_to_x_(R, inds_ABCD[i], concise_ladJ = concise_ladJ) # (m,3), (m,1)
            X_mixed.append(x) ; ladJ_RX += ladJ
        else:
            X_mixed.append(R[:,i])
    X_mixed = tf.stack(X_mixed, axis=1) # (m,n,3)
    return X_mixed, ladJ_RX       # (m,n,3), (m,1)

def X_to_R_from_origin_tf_(X,
                           inds_ABCD : list,
                           reconstriction_sequence : list,
                           index_of_starting_distance : int,
                           concise_ladJ = True,
                           ):
    # X : (m,n,3) ; m = # frame ; n = # atoms ; 3 IC variable [b,a,t]
    # ABCD ~ (n,4)
    # reconstriction_sequence : (n,) : order in which ABCD is used to build molecule
    # index_of_starting_distance : int 

    distances = X[:,:,0] # (m,n)
    angles =    X[:,:,1] # (m,n)
    torsions =  X[:,:,2] # (m,n)

    R = [] ; ladJ_XR = 0.0
    
    A, B, C, D = inds_ABCD[reconstriction_sequence[0]]

    dCB = distances[:, index_of_starting_distance:index_of_starting_distance + 1] # (m,1)
    
    rB = tf.zeros_like(X[:,0,:])              # (m,3)
    rC = tf.concat( [dCB, rB[:,:2]], axis=-1) # (m,3)
    rD = rC + tf.constant( [[0.,10.,0.]] )    # (m,3)
    
    R.append(rB) ; R.append(rC) ; permutation = [B,C]
    
    rA, ladJ = NeRF_tf_(distances[:,A], # (m,)
                        angles[:,A],    # (m,)
                        torsions[:,A],  # (m,)
                        rB,             # (m,3)
                        rC,             # (m,3)
                        rD,             # (m,3)
                        dont_return_ladJ = concise_ladJ,
                        )               # outputs ~ (m,3), (m,1) or (,)

    R.append(rA) ; permutation.append(A) ; ladJ_XR += ladJ
    
    for A in reconstriction_sequence[1:]:
        A, B, C, D = inds_ABCD[A]

        rB = tf.gather(R, tf.where(tf.equal(permutation, B))[0][0]) # (m,3)
        rC = tf.gather(R, tf.where(tf.equal(permutation, C))[0][0]) # (m,3)
        rD = tf.gather(R, tf.where(tf.equal(permutation, D))[0][0]) # (m,3)
        
        rA, ladJ = NeRF_tf_(distances[:,A], # (m,)
                            angles[:,A],     # (m,)
                            torsions[:,A],   # (m,)
                            rB,              # (m,3)
                            rC,              # (m,3)
                            rD,              # (m,3)
                            dont_return_ladJ = concise_ladJ,
                            )                # outputs ~ (m,3), (m,3,3)

        R.append(rA) ; permutation.append(A) ; ladJ_XR += ladJ

    R = tf.stack(R)                                   # (n, m, 3)
    R = tf.gather(R, tf.argsort(permutation), axis=0) # (n, m, 3)
    R = tf.einsum('ijk->jik', R)                      # (m, n, 3)

    if concise_ladJ: ladJ_XR -= R_to_X_(R, inds_ABCD, concise_ladJ = True)[-1]
    else: pass
    return R, ladJ_XR # (m,n,3), (m,1)

def X_mixed_to_R_tf_(X_mixed,           # (m,n,3) in the correct order
                     inds_ABCD : list , # (n,4)
                     inds_IC : list,    # (n_IC,)
                     inds_XYZ : list,   # (n_XYZ,)
                     reconstriction_sequence : list, # (n_IC)
                     concise_ladJ = True,
                     ):
    X = tf.gather(X_mixed,inds_IC,axis=1)        ; n_IC  = X.shape[1]
    R_seeds = tf.gather(X_mixed,inds_XYZ,axis=1) ; n_XYZ = R_seeds.shape[1]

    distances = X[:,:,0] # (m,nIC)
    angles =    X[:,:,1] # (m,nIC)
    torsions =  X[:,:,2] # (m,nIC)

    R = [R_seeds[:,i,:] for i in range(n_XYZ)] ; permutation = list(inds_XYZ)
    ladJ_XR = 0.0

    for A in reconstriction_sequence:
        A, B, C, D = inds_ABCD[A]

        _A = tf.where(tf.equal(inds_IC, A))[0][0]

        rB = tf.gather(R, tf.where(tf.equal(permutation, B))[0][0]) # (m,3)
        rC = tf.gather(R, tf.where(tf.equal(permutation, C))[0][0]) # (m,3)
        rD = tf.gather(R, tf.where(tf.equal(permutation, D))[0][0])   # (m,3) 

        rA, ladJ = NeRF_tf_(distances[:,_A], angles[:,_A], torsions[:,_A], rB, rC, rD, dont_return_ladJ = concise_ladJ)
        R.append(rA) ; permutation.append(A) ; ladJ_XR += ladJ

    R = tf.stack(R)                                   # (n, m, 3)
    R = tf.gather(R, tf.argsort(permutation), axis=0) # (n, m, 3)
    R = tf.einsum('ijk->jik', R)                      # (m, n, 3)

    if concise_ladJ: ladJ_XR -= R_to_X_mixed_(R, inds_ABCD=inds_ABCD, inds_IC=inds_IC, concise_ladJ = True)[-1]
    else: pass
    return R, ladJ_XR # (m,n,3), (m,1)

##

class CONSTANT_SCALE_RESHAPE:
    def __init__(self,
                 X, # data
                 inds_IC : list,
                 inds_XYZ : list, #tunnel_mask = None, #np.array; needed if inds_XYZ = []
                 eps_singularity : list = [1e-2, 1e-5],
                 model_range = [-1.0, 1.0],
                 general_ranges = False, # not implemented in this version.
                 ):
        self.general_ranges = False,
         
        self.model_range = model_range
        self.n_atoms = len(inds_IC) + len(inds_XYZ)
        self.dim_max = self.n_atoms*3
        self.inds_IC = inds_IC
        self.inds_XYZ = inds_XYZ
        self.eps_static = eps_singularity[1]
        self.eps_istropic = eps_singularity[0]
        ##
        self.MIN_model = self.model_range[0] # (,)
        self.MAX_model = self.model_range[1] # (,)
        self.RANGE_model = self.MAX_model - self.MIN_model # (,)
        ##
        complete_variable_types_mask = np.array([['b','a','t']]*self.n_atoms) 
        for i in inds_XYZ:
            complete_variable_types_mask[i] = 'c'
        complete_variable_types_mask = complete_variable_types_mask.flatten()
        ##
        petriodic_mask = np.array([[0,0,1]]*self.n_atoms) # (n_atoms,3)
        angle_mask = np.array([[0,1,0]]*self.n_atoms)     # (n_atoms,3)
        bond_mask = np.array([[1,0,0]]*self.n_atoms)      # (n_atoms,3)
        xyz_mask = np.array([[0,0,0]]*self.n_atoms) 
        for i in inds_XYZ:
            petriodic_mask[i] = 0 ; angle_mask[i] = 0 ; bond_mask[i] = 0 ; xyz_mask[i] = 1
        self.xyz_mask_flat = xyz_mask.flatten()
        self.PM= petriodic_mask.flatten()

        X = np.array(X).astype(np.float64)
        self.MU_physical = X.mean(0,keepdims=True)
        self.SD_physical = np.sqrt(((X-self.MU_physical)**2).mean(0,keepdims=True))
        self.MU_physical = self.MU_physical.astype(np.float32).flatten()
        self.SD_physical = self.SD_physical.astype(np.float32).flatten()
        self.MIN_physical = X.min(0).astype(np.float32).flatten() # (dim_max,)
        self.MAX_physical = X.max(0).astype(np.float32).flatten() # (dim_max,)
        self.MIN_physical[np.where(self.PM==1)[0]] = -PI
        self.MAX_physical[np.where(self.PM==1)[0]] =  PI
        self.RANGE_physical = self.MAX_physical - self.MIN_physical # (dim_max,)

        self.mask_static = np.where(self.RANGE_physical <= self.eps_static, 1, 0) # (dim_max,)
        self.mask_isotropic = np.where((self.RANGE_physical > self.eps_static)&(self.RANGE_physical <= self.eps_istropic), 1, 0)
        self.mask_flowing = np.ones([self.dim_max]).astype(np.int32) - self.mask_static - self.mask_isotropic
        if min(self.mask_flowing) <0: print('!! masks')
        else: pass
        self.n_static = self.mask_static.sum()
        self.n_isotropic = self.mask_isotropic.sum()
        self.n_flowing = self.mask_flowing.sum()
        if self.n_static + self.n_isotropic + self.n_flowing != self.dim_max: print('!!!! masks')
        else: pass
        print('# variables static:   ', self.n_static)
        print('# variables isotropic:', self.n_isotropic)
        print('# variables flowing:  ', self.n_flowing )
        print('# total variables', self.n_flowing + self.n_isotropic + self.n_static)
        
        self.inds_static = np.where(self.mask_static == 1)[0]#.tolist()
        self.inds_isotropic = np.where(self.mask_isotropic == 1)[0]#.tolist()
        self.inds_flowing = np.where(self.mask_flowing == 1)[0]#.tolist()

        cat_inds = np.array(self.inds_static.tolist() + self.inds_isotropic.tolist() + self.inds_flowing.tolist())
        self.inds_gather_inverse = np.array([int(np.where(cat_inds == i)[0]) for i in range(len(cat_inds))]).tolist()

        self.periodic_mask = self.PM[self.inds_flowing].tolist()
        self.variable_types_mask = complete_variable_types_mask[self.inds_flowing].tolist()

        self.MU_static = clamp_float_tf_(self.MU_physical[self.inds_static][np.newaxis,:])

        self.MU_isotropic = clamp_float_tf_(self.MU_physical[self.inds_isotropic][np.newaxis,:])
        self.SD_isotropic = clamp_float_tf_(self.SD_physical[self.inds_isotropic][np.newaxis,:])
        self.MIN_isotropic = clamp_float_tf_(self.MIN_physical[self.inds_isotropic][np.newaxis,:])
        self.MAX_isotropic = clamp_float_tf_(self.MAX_physical[self.inds_isotropic][np.newaxis,:])

        self.MIN_flowing = clamp_float_tf_(self.MIN_physical[self.inds_flowing][np.newaxis,:])
        self.MAX_flowing = clamp_float_tf_(self.MAX_physical[self.inds_flowing][np.newaxis,:])
        self.RANGE_flowing = clamp_float_tf_(self.RANGE_physical[self.inds_flowing][np.newaxis,:])

        self.gaussian = tfp.distributions.TruncatedNormal(
                            loc = self.MU_isotropic[0],
                            scale = self.SD_isotropic[0],
                            low = self.MIN_isotropic[0],
                            high = self.MAX_isotropic[0])

        self.scale_forward = self.RANGE_model / self.RANGE_flowing  ; self.scale_inverse = 1.0 / self.scale_forward
        self.ladJ_forward = tf.reduce_sum( tf.math.log(self.scale_forward) ) ; self.ladJ_inverse = - self.ladJ_forward

    def forward(self, X):
        #m = X.shape[0]
        m = tf.shape(X)[0]
        X_flat = tf.reshape(X, [m,self.dim_max])
        
        X_flat_flowing = tf.gather(X_flat, self.inds_flowing, axis=-1)

        X_flat_flowing_model = self.scale_forward*(X_flat_flowing - self.MIN_flowing) + self.MIN_model

        X_flat_isotropic = tf.gather(X_flat, self.inds_isotropic, axis=-1)
        ln_base = tf.reduce_sum(self.gaussian.log_prob(X_flat_isotropic),axis=-1,keepdims=True)

        return X_flat_flowing_model , self.ladJ_forward + ln_base


    def inverse(self, X_flat_flowing_model):
        #m = X_flat_flowing_model.shape[0]
        m = tf.shape(X_flat_flowing_model)[0]

        X_flat_flowing = self.scale_inverse*(X_flat_flowing_model - self.MIN_model) + self.MIN_flowing
        
        X_flat_isotropic = self.gaussian.sample(m)
        ln_base = tf.reduce_sum(self.gaussian.log_prob(X_flat_isotropic),axis=-1,keepdims=True)

        X_flat_static = self.MU_static + tf.zeros([m,self.n_static])

        X_flat = tf.gather(tf.concat([X_flat_static, X_flat_isotropic, X_flat_flowing],axis=-1),
                                    self.inds_gather_inverse, axis=-1)
        X = tf.reshape(X_flat, [m, self.n_atoms, 3])

        return X, self.ladJ_inverse - ln_base
    
    def save_this_(self,name):
        save_pickle_([self.scale_forward, self.scale_inverse, self.MIN_flowing, self.ladJ_forward, self.MU_static, self.ladJ_inverse ],
                     name)
    def overwrite_this_(self,name):
        self.scale_forward, self.scale_inverse, self.MIN_flowing, self.ladJ_forward, self.MU_static, self.ladJ_inverse  = load_pickle_(name)

##
import pickle
def save_pickle_(x,name):
    with open(name, "wb") as f: pickle.dump(x, f) ; print('saved',name)
    
def load_pickle_(name):
    with open(name, "rb") as f: x = pickle.load(f) ; return x
##

def try_get_conditioned_reconstriction_sequence_(inds_placed: list, 
                                                 ABCD : list, # (n,4)
                                                 verbose : bool = False):

    # inds_placed : [A,B,C] from ABCD[A], in case of full IC trasformation.
    #               inds_XYZ, in case of partial (mixed) IC trasformation.
    # ABCD ~ (n,4) ; n = # atoms 

    n_atoms = len(ABCD) #.shape[0]

    placed_set = set(inds_placed)
    not_placed_set = list( set(np.arange(n_atoms)) - placed_set )

    reconstriction_sequence = []
    while len(placed_set) != n_atoms:
        placed_something = False
        for A in not_placed_set:
            if A not in placed_set:
                BCD = ABCD[A][1:] 
                if set(BCD).issubset(placed_set):
                    reconstriction_sequence.append(A)
                    placed_set |= set([A])
                    placed_something = True
        if placed_something is False:
            break
        else: pass
    if placed_something:
        if verbose: print('These indices work. All atoms can be reached.')
        else: pass
        return reconstriction_sequence
    else: 
        if verbose: print('!! Invalid indices provided (some atoms can not be reached).')
        else: pass    
        return None

###########################################################################

def depth_first_search_(graph, node):
    # Reference:  https://www.educative.io/edpresso/how-to-implement-depth-first-search-in-python
    nodes = set() ; visited = set()
    def dfs(visited, graph, node):
        if node not in visited:
            nodes.add(node)
            visited.add(node)
            for neighbour in graph[node]:
                dfs(visited, graph, neighbour)
    dfs(visited, graph,node)
    return nodes 

def check_neighbours_ranks_(me, am):
    ranks = []
    for neighbour in range(len(am)):
        if am[me,neighbour] > 0:
            am_cut = np.array(am)
            am_cut[me,neighbour] = 0 ; am_cut[neighbour,me] = 0
            graph = {i: np.nonzero(row)[0].tolist() for i,row in enumerate(am_cut)}
            ranks.append(len(depth_first_search_(graph,neighbour)))
        else: ranks.append(0)
    return np.array(ranks)

def get_neighbour_lists_(am):
    DCBA = []
    for D in range(am.shape[0]):
        am_cut = np.array(am)
        C = np.argmax(am_cut[D]*check_neighbours_ranks_(D,am)) ; am_cut[C,D] = 0 ; am_cut[D,C] = 0
        B = np.argmax(am_cut[C]*check_neighbours_ranks_(C,am)) ; am_cut[B,C] = 0 ; am_cut[C,B] = 0
        A = np.argmax(am_cut[B]*check_neighbours_ranks_(B,am)) ; am_cut[A,B] = 0 ; am_cut[B,A] = 0
        DCBA.append([D,C,B,A])
    return np.array(DCBA) # (n,4)

##########################################################################

class XR_MAP_mixed_only:

    def automatic_search_for_stA_and_stBC_(self):
        yes = False
        ABCD = np.array(self.ABCD)
        for A in range(self.n_atoms):
            B, C =  ABCD[A,[1,2]]
            for i in range(self.n_atoms):
                if set([B,C]).issubset(ABCD[i,:2]):
                    yes = True
                    break
                else: pass
            if yes:
                reconstriction_sequence = try_get_conditioned_reconstriction_sequence_(ABCD[A][:3], self.ABCD)
                if reconstriction_sequence is not None: break
                else: pass
            else: pass

        if reconstriction_sequence is not None:
            self.reconstriction_sequence = [A] + reconstriction_sequence 
            self.stBC = i
            self.stA = A  # only for information, not used later.
            return True
        else: 
            print('\n !!! : There was a problem.')
            return False


    def find_6_redundant_dof_in_full_IC_map_(self,this):
        mask = np.zeros([self.n_atoms,3]).astype(np.int32) # (n,3)
        A,B,C,D = self.ABCD[self.stA]
        mask[C]       = -1  # the origin                     # - 3 d.o.f. 
        mask[B,[1,2]] = -1  # only need to know the bond [0] # - 2 d.o.f. 
        mask[D,2]     = -2*this  # torsion from,                                            # these signs
        mask[A,2]     =  2*this  # to.                            # - 1 d.o.f.              # were wrong way round.
        self.tunnel_mask = mask

    def __init__(self,
                 PDB, # of a single molecule in vacuum.
                 ABCD_method = 0, # one of them does not work.
                 model_range = [-1.0,1.0]
                ):
        """
        1
        """
        self.model_range = model_range
        self.mol = Chem.MolFromPDBFile(PDB, removeHs = False) # obj
        for i, a in enumerate(self.mol.GetAtoms()): a.SetAtomMapNum(i)
        self.masses = np.array([x.GetMass() for x in self.mol.GetAtoms()]) # (n_atoms,)
        self.inds_hydrogens = np.where(self.masses<1.009)[0]
        self.n_hydrogens = self.inds_hydrogens.shape[0]

        self.adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix( self.mol ) # (n_atoms,n_atoms)
        self.n_atoms = self.adjacency_matrix.shape[0]

        if ABCD_method == 0:
            self.ABCD = get_neighbour_lists_(self.adjacency_matrix).tolist()
        else:
            print('method removed until fixed')
            #self.ABCD = get_ABCD_v3_(self.adjacency_matrix).tolist()

        #'''
        if self.automatic_search_for_stA_and_stBC_():
            print('full IC would be possible')
            print('next run align_data_(R_data)')
            '''
            self.n_IC = int(self.n_atoms) ; self.inds_IC = np.arange(self.n_atoms).tolist()
            self.n_XYZ = 0 ; self.inds_XYZ = None

            self.CSR_obj = CONSTANT_SCALE_RESHAPE(X = None,
                                                  inds_IC = self.inds_IC,
                                                  inds_XYZ = [],
                                                  eps_singularity = 1e-3,
                                                  model_range = model_range,
                                                  verbose = True,
                                                 )
            '''
        else: print('try second ABCD_method or initialise mixed transfromation')
        #'''
    
    def align_data_(self, Rdata, Inds_keep_cartesian=None, general_ranges = False, eps_singularity = [1e-6,1e-6]):
        Rdata = clamp_float_tf_(Rdata)
        Xdata = R_to_X_(Rdata, inds_ABCD = self.ABCD)[0]
        _Rdata = X_to_R_from_origin_tf_(Xdata,
                                inds_ABCD = self.ABCD,
                                reconstriction_sequence = self.reconstriction_sequence,
                                index_of_starting_distance = self.stBC,
                                )[0]
        try:
            self.find_6_redundant_dof_in_full_IC_map_(1.0)
            inds_keep_cartesian = np.argsort(np.where(self.tunnel_mask==-2,-1,self.tunnel_mask).sum(1))[:3]
            print('inds_keep_cartesian:',inds_keep_cartesian)
            if Inds_keep_cartesian is None: _inds_keep_cartesian = inds_keep_cartesian 
            else: _inds_keep_cartesian = Inds_keep_cartesian
            self.set_which_atoms_to_be_kept_cartisian_(_inds_keep_cartesian)
            self.show_this_object_some_data_(_Rdata, general_ranges=general_ranges, eps_singularity=eps_singularity)
        except:
            self.find_6_redundant_dof_in_full_IC_map_(-1.0)
            inds_keep_cartesian = np.argsort(np.where(self.tunnel_mask==-2,-1,self.tunnel_mask).sum(1))[:3]
            print('inds_keep_cartesian:',inds_keep_cartesian)
            if Inds_keep_cartesian is None: _inds_keep_cartesian = inds_keep_cartesian 
            else: _inds_keep_cartesian = Inds_keep_cartesian
            self.set_which_atoms_to_be_kept_cartisian_(_inds_keep_cartesian)
            self.show_this_object_some_data_(_Rdata, general_ranges=general_ranges, eps_singularity=eps_singularity)

        return _Rdata.numpy()

    def set_which_atoms_to_be_kept_cartisian_(self, inds_keep_these_atoms_cartesian : list):
        """
        2
        """
        inds_keep_these_atoms_cartesian = sorted(inds_keep_these_atoms_cartesian)

        reconstriction_sequence = try_get_conditioned_reconstriction_sequence_(inds_keep_these_atoms_cartesian, 
                                                                               ABCD = self.ABCD,
                                                                               verbose = True)
        if reconstriction_sequence is not None:
            self.inds_XYZ = np.array(inds_keep_these_atoms_cartesian).flatten()
            self.n_XYZ = self.inds_XYZ.shape[0]
            self.inds_IC = np.array(list( set(np.arange(self.n_atoms)) - set(self.inds_XYZ) )).flatten().tolist()
            self.n_IC = len(self.inds_IC)
            self.reconstriction_sequence = reconstriction_sequence
            #print('next run show_this_object_some_data_(aligned_data) before using')

        else:
            print('Please try a different set.')
            return None

    def show_this_object_some_data_(self, Rdata, general_ranges = False, eps_singularity=[1e-6,1e-6]):
        # general_ranges = False is fixed.
        """
        3
        """
        Rdata = clamp_float_tf_(Rdata)
        # data should be aligned to identity referece frame by a planar subset.
        Xdata = R_to_X_mixed_(Rdata, inds_ABCD = self.ABCD, inds_IC=self.inds_IC, concise_ladJ = True)[0]
        self.CSR_obj = CONSTANT_SCALE_RESHAPE(  X = Xdata,
                                                inds_IC = self.inds_IC,
                                                inds_XYZ = self.inds_XYZ,
                                                eps_singularity = eps_singularity,
                                                model_range = self.model_range,
                                                general_ranges = general_ranges,
                                            )
        
    def forward(self, R, concise_ladJ=True):
        X, ladJ_RX = R_to_X_mixed_(R, inds_ABCD = self.ABCD, inds_IC=self.inds_IC, concise_ladJ = concise_ladJ)
        x, ladJ_Xx = self.CSR_obj.forward(X)
        return x, ladJ_RX + ladJ_Xx
    
    def inverse(self, x, concise_ladJ=True):
        X, ladJ_xX = self.CSR_obj.inverse(x)
        R, ladJ_XR = X_mixed_to_R_tf_(  X_mixed = X,
                                        inds_ABCD = self.ABCD,
                                        inds_IC = self.inds_IC,
                                        inds_XYZ = self.inds_XYZ,
                                        reconstriction_sequence = self.reconstriction_sequence,
                                        concise_ladJ = concise_ladJ,
                                    )
        return R, ladJ_XR + ladJ_xX

    ##
    '''
    def forward(self, R, concise_ladJ=True):
        x, ladJ = self.forward_(R, concise_ladJ = concise_ladJ)
        #return x, ladJ
        return forward_this_(x), ladJ
    
    def inverse(self, x, concise_ladJ=True):
        x = inverse_this_(x, signs = [1.0,1.0,1.0])

        R, ladJ = self.inverse_(x, concise_ladJ = concise_ladJ)
        return R, ladJ
    '''
''' # keeping only one trainable hydrogen torsion in each of the 3 methyl groups in AD 
shift_up_by_third_01_ = lambda x, sign=1.0 : tf.math.floormod(x+1.0+sign*2./3.,2.0)-1.0

inds_forward_gather = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46]

def forward_this_(X):
    return tf.gather(X, inds_forward_gather, axis=-1)

def inverse_this_(X, signs = [1.0,1.0,1.0]):

    tAfro = X[:,1:2]
    tAto1 = shift_up_by_third_01_(tAfro, sign=signs[0])
    tAto2 = shift_up_by_third_01_(tAto1, sign=signs[0])

    tBfro = X[:,21:22]
    tBto1 = shift_up_by_third_01_(tBfro, sign=signs[1])
    tBto2 = shift_up_by_third_01_(tBto1, sign=signs[1])

    tCfro = X[:,39:40]
    tCto1 = shift_up_by_third_01_(tCfro, sign=signs[2])
    tCto2 = shift_up_by_third_01_(tCto1, sign=signs[2])

    return tf.concat([X[:,:6],tAto1,X[:,6:7],tAto2,
                      X[:,7:23],tBto1,X[:,23:24],tBto2,
                      X[:,24:41],tCto1,X[:,41:42],tCto2],axis=-1)
'''
