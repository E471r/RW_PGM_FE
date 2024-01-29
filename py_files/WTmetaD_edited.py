# ! Reference: https://web.stanford.edu/~peastman/metadynamics.py
'''

This whole .py file was obtained from https://web.stanford.edu/~peastman/metadynamics.py

The implementation was *edited slightly compared to original. 
The current version is less general than original. *TODO: edit properly.

'''

import simtk.openmm as mm
import simtk.unit as unit
import numpy as np
from functools import reduce

class wtmetad(object):
    """Performs metadynamics.

    This class implements well-tempered metadynamics, as described in Barducci et al.,
    "Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method"
    (https://doi.org/10.1103/PhysRevLett.100.020603).  You specify from one to three
    collective variables whose sampling should be accelerated.  A biasing force that
    depends on the collective variables is added to the simulation.  Initially the bias
    is zero.  As the simulation runs, Gaussian bumps are periodically added to the bias
    at the current location of the simulation.  This pushes the simulation away from areas
    it has already explored, encouraging it to sample other regions.  At the end of the
    simulation, the bias function can be used to calculate the system's free energy as a
    function of the collective variables.

    To use the class you create a Metadynamics object, passing to it the System you want
    to simulate and a list of BiasVariable objects defining the collective variables.
    It creates a biasing force and adds it to the System.  You then run the simulation
    as usual, but call step() on the Metadynamics object instead of on the Simulation.
    """

    def __init__(self, system, variables, temperature, deltaT, height, frequency):
        """Create a Metadynamics object.

        Parameters
        ----------
        system: System
            the System to simulate.  A CustomCVForce implementing the bias is created and
            added to the System.
        variables: list of BiasVariables
            the collective variables to sample
        temperature: temperature
            the temperature at which the simulation is being run.  This is used in computing
            the free energy.
        deltaT: temperature
            the temperature offset used in scaling the height of the Gaussians added to the
            bias.  The collective variables are sampled as if the effective temperature of
            the simulation were temperature+deltaT.
        height: energy
            the initial height of the Gaussians to add
        frequency: int
            the interval in time steps at which Gaussians should be added to the bias potential
        """
        if not unit.is_quantity(temperature):
            temperature = temperature*unit.kelvin
        if not unit.is_quantity(deltaT):
            deltaT = deltaT*unit.kelvin
        if not unit.is_quantity(height):
            height = height*unit.kilojoules_per_mole
        self.variables = variables
        self.temperature = temperature
        self.deltaT = deltaT
        self.height = height
        self.frequency = frequency
        self._bias = np.zeros(tuple(v.gridWidth for v in variables))
        varNames = ['cv%d' % i for i in range(len(variables))]
        self._force = mm.CustomCVForce('table(%s)' % ', '.join(varNames))
        for name, var in zip(varNames, variables):
            self._force.addCollectiveVariable(name, var.force)
        widths = [v.gridWidth for v in variables]
        mins = [v.minValue for v in variables]
        maxs = [v.maxValue for v in variables]
        if len(variables) == 1:
            self._table = mm.Continuous1DFunction(self._bias.flatten(), mins[0], maxs[0])
        elif len(variables) == 2:
            self._table = mm.Continuous2DFunction(widths[0], widths[1], self._bias.flatten(), mins[0], maxs[0], mins[1], maxs[1], periodic=True)
            self.grids = [np.linspace(0.0,1.0,n) for n in widths]

        elif len(variables) == 3:
            self._table = mm.Continuous3DFunction(widths[0], widths[1], widths[2], self._bias.flatten(), mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
        else:
            raise ValueError('Metadynamics requires 1, 2, or 3 collective variables')
        self._force.addTabulatedFunction('table', self._table)
        self._force.setForceGroup(15)
        system.addForce(self._force)

        self.kT = self.temperature._value*1e-3*8.31446261815324

    def step(self, simulation, steps):
        """Advance the simulation by integrating a specified number of time steps.

        Parameters
        ----------
        simulation: Simulation
            the Simulation to advance
        steps: int
            the number of time steps to integrate
        """

        self.list_CV = []
        self.list_coordinates = []
        self.list_u = []
        self.list_v = []
        self.list_K_u = []
        self.list_K_v = []

        stepsToGo = steps

        while stepsToGo > 0:
            nextSteps = stepsToGo
            if simulation.currentStep % self.frequency == 0:
                nextSteps = min(nextSteps, self.frequency)
            else:
                nextSteps = min(nextSteps, simulation.currentStep % self.frequency)
            simulation.step(nextSteps)
            if simulation.currentStep % self.frequency == 0:
                position = self._force.getCollectiveVariableValues(simulation.context)
                self.list_CV.append(position)
                V = simulation.context.getState(getEnergy=True, groups={15}).getPotentialEnergy()
                height = self.height*np.exp(-V/(unit.MOLAR_GAS_CONSTANT_R*self.deltaT))
                self._addGaussian(position, height, simulation.context)

                ##
                ## unit=kilojoule/mole ##
                v = V._value / self.kT
                xyz = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
                u = simulation.context.getState(getEnergy=True, groups={0}).getPotentialEnergy()._value / self.kT
                K_u = simulation.context.getState(getEnergy=True, groups={0}).getKineticEnergy()._value
                K_v = simulation.context.getState(getEnergy=True, groups={15}).getKineticEnergy()._value

                self.list_coordinates.append(xyz)
                self.list_u.append(u)
                self.list_v.append(v)
                self.list_K_u.append(K_u)
                self.list_K_v.append(K_v)

                ##
    
            stepsToGo -= nextSteps
    
    def _step(self, simulation, steps):
        self.hist_s = []

        for i in range(steps//self.frequency):
            simulation.step(self.frequency-1)
            position = self._force.getCollectiveVariableValues(simulation.context)
            
            self.hist_s.append(position)
            
            energy = simulation.context.getState(getEnergy=True, groups={15}).getPotentialEnergy()
            height = self.height*np.exp(-energy/(unit.MOLAR_GAS_CONSTANT_R*self.deltaT))
            self._addGaussian(position, height, simulation.context)
            simulation.step(1)
            

    def getFreeEnergy(self):
        """Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number of collective
        variables.  The values are in kJ/mole.  The i'th position along an axis corresponds to
        minValue + i*(maxValue-minValue)/gridWidth.
        """
        return -((self.temperature+self.deltaT)/self.deltaT)*self._bias

    def _addGaussian(self, position, height, context):
        """Add a Gaussian to the bias function."""
        # Compute a Gaussian along each axis.

        axisGaussians = []
        for i,v in enumerate(self.variables):
            x = (position[i]-v.minValue) / (v.maxValue-v.minValue)
            if v.periodic:
                x = x % 1.0
            dist = np.abs(self.grids[i]- x)
            #if v.periodic:
            dist = np.min(np.array([dist, np.abs(dist-1)]), axis=0)
            gaus = np.exp(-dist*dist*v.gridWidth/v.biasWidth)
            gaus[0] = gaus[-1]
            axisGaussians.append(gaus)

        # Compute their outer product.

        if len(self.variables) == 1:
            gaussian = axisGaussians[0]
        else:
            gaussian = reduce(np.multiply.outer, reversed(axisGaussians))

        # Add it to the bias.

        height = height.value_in_unit(unit.kilojoules_per_mole)
        self._bias += height*gaussian
        widths = [v.gridWidth for v in self.variables]
        mins = [v.minValue for v in self.variables]
        maxs = [v.maxValue for v in self.variables]
        if len(self.variables) == 1:
            self._table.setFunctionParameters(self._bias.flatten(), mins[0], maxs[0])
        elif len(self.variables) == 2:
            self._table.setFunctionParameters(widths[0], widths[1], self._bias.flatten(), mins[0], maxs[0], mins[1], maxs[1])
        elif len(self.variables) == 3:
            self._table.setFunctionParameters(widths[0], widths[1], widths[2], self._bias.flatten(), mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
        self._force.updateParametersInContext(context)


class BiasVariable(object):
    """A collective variable that can be used to bias a simulation with metadynamics."""

    def __init__(self, force, minValue, maxValue, biasWidth, periodic=False, gridWidth=None):
        """Create a BiasVariable.

        Parameters
        ----------
        force: Force
            the Force object whose potential energy defines the collective variable
        minValue: float
            the minimum value the collective variable can take.  If it should ever go below this,
            the bias force will be set to 0.
        maxValue: float
            the maximum value the collective variable can take.  If it should ever go above this,
            the bias force will be set to 0.
        biasWidth: float
            the width (standard deviation) of the Gaussians added to the bias during metadynamics
        periodic: bool
            whether this is a periodic variable, such that minValue and maxValue are physical equivalent
        gridWidth: int
            the number of grid points to use when tabulating the bias function.  If this is omitted,
            a reasonable value is chosen automatically.
        """
        self.force = force
        self.minValue = minValue
        self.maxValue = maxValue
        self.biasWidth = biasWidth
        self.periodic = periodic
        if gridWidth is None:
            self.gridWidth = int(np.ceil(5*(maxValue-minValue)/biasWidth))
        else:
            self.gridWidth = gridWidth


"""
def get_grid_01_(n_bins, tight=True):
    if tight:
        return np.linspace(0.0, 1.0, n_bins)
    else:
        return np.linspace(0.0 + 0.5/n_bins, 1.0 - 0.5/n_bins, n_bins)

class BV(object):
    def __init__(self,
                 force,
                 sigma,
                 n_bins,
                 _range = [-np.pi,np.pi]
                 ):
        self.force = force
        self.sigma = sigma
        self.sigma_squared = sigma**2
        self.n_bins = n_bins
        self.MIN, self.MAX = _range
        self.domain_width = self.MAX - self.MIN
        self.grid = get_grid_01_(n_bins)*self.domain_width + self.MIN
        self.periodic = True

#
#import simtk.unit as unit
#unit.MOLAR_GAS_CONSTANT_R
#Quantity(value=8.31446261815324, unit=joule/(kelvin*mole))

import simtk.openmm as mm
import simtk.unit as unit
import numpy as np
from functools import reduce
# joules_per_mole = (1.0*unit.joules)/(1.0*mm.unit.mole)

class WTmetaD(object):
    def __init__(self,
                 system,
                 CVs,
                 T, # K but dont include unit!
                 gamma, # height of barrier of interest in kT (no unit!)
                 h0, # unit=None/kT
                 stride,
                 ):
        kB = unit.MOLAR_GAS_CONSTANT_R      # unit=joule/(kelvin*mole)
        T = T*unit.kelvin                   # unit=kelvin
        kT = kB*T                           # unit=joule/mole
        deltaT = gamma*T - T                # unit=kelvin
        kdeltaT = kB*deltaT                 # unit=joule/mole
        WT_constant = - (T + deltaT)/deltaT # unit=None/kT
        h0 = h0*unit.kilojoules_per_mole    # unit=kilojoules/mole

        # V = simulation.context.getState(getEnergy=True, groups={15}).getPotentialEnergy()
        # ^ unit=unit=kilojoule/mole
        # V/kdeltaT # unit=kT

        # h0 # unit=kT
        # ht = h0 * exp(-V/kdeltaT)         # unit=kT
        # V = ht * KDE                      # unit=kilojoules/mole
        # F = WT_constant*V                 # unit=kilojoules/mole

        self.n_CVs = len(CVs)
        self.CVs = CVs

        self.kB = kB
        self.T = T
        self.kT = kT
        self.gamma = gamma
        self.deltaT = deltaT
        self.kdeltaT = kdeltaT
        self.WT_constant = WT_constant
        self.h0 = h0
        self.stride = stride

        names = ['CV_%d' % i for i in range(self.n_CVs)]
        self.list_sigma_squared = [v.sigma_squared for v in CVs]
        self.MINs = [x.MIN for x in CVs]
        self.MAXs = [x.MAX for x in CVs]
        self.domain_widths = [x.domain_width for x in CVs]
        self.grids = [x.grid for x in CVs]
        self.list_n_bins = [v.n_bins for v in CVs]
        self._bias = np.zeros(self.list_n_bins)
        self._force = mm.CustomCVForce('table(%s)' % ', '.join(names)) # 'table(CV_0, CV_1, ...)'
        for name, x in zip(names, CVs):
            self._force.addCollectiveVariable(name, x.force)
        if self.n_CVs == 1:
            self._table = mm.Continuous1DFunction(self._bias.flatten(),
                                                  self.MINs[0], self.MAXs[0],
                                                  periodic = True,
                                                  )
        elif self.n_CVs == 2:
            self._table = mm.Continuous2DFunction(self.list_n_bins[0], self.list_n_bins[1],
                                                  self._bias.flatten(), 
                                                  self.MINs[0], self.MAXs[0], self.MINs[1], self.MAXs[1],
                                                  periodic = True,
                                                  )
        else:
            raise ValueError('use only 1 or 2 collective variables')
        self._force.addTabulatedFunction('table', self._table)
        self._force.setForceGroup(15)
        system.addForce(self._force)

        self.reset_log()
        
    def reset_log(self):
        self.ss = []

    def step_normally(self, simulation, n_windows, stride):
        for i in range(n_windows):
            simulation.step(stride)
            position = self._force.getCollectiveVariableValues(simulation.context)
            self.ss.append(position)
    @property
    def get_log(self):
        return np.array(self.ss)

    '''
    def step(self, simulation, steps):
        stepsToGo = steps
        while stepsToGo > 0:
            nextSteps = stepsToGo
            if simulation.currentStep % self.stride == 0:
                nextSteps = min(nextSteps, self.stride)
            else:
                nextSteps = min(nextSteps, simulation.currentStep % self.stride)
            simulation.step(nextSteps)
            if simulation.currentStep % self.stride == 0:
                position = self._force.getCollectiveVariableValues(simulation.context)
                V = simulation.context.getState(getEnergy=True, groups={15}).getPotentialEnergy()
                h = self.h0*np.exp(-V/self.kdeltaT)
                self._addGaussian(position, h, simulation.context)
                self.st = np.array(position)
            stepsToGo -= nextSteps

    @property
    def FES(self):
        fes = self.WT_constant*self._bias #/ self.kT
        return fes - fes.min()

    def _addGaussian(self, s, h, context):
        
        ws = []
        for i, x in enumerate(self.CVs):
            x = s[i]
            grid = self.grids[i]
            x_min = self.MINs[i]
            domain_width = self.domain_widths[i]
            sigma_squared = self.list_sigma_squared[i]
            dist_sq = (np.mod(x - grid - x_min,domain_width) + x_min)**2
            hill = np.exp(-dist_sq/sigma_squared)
            ws.append(hill)

        if self.n_CVs == 1: gaussian = ws[0]
        else:               gaussian = reduce(np.multiply.outer, reversed(ws))

        h = h.value_in_unit(unit.kilojoules_per_mole)
        self._bias += h*gaussian

        if self.n_CVs == 1:
            self._table = mm.Continuous1DFunction(self._bias.flatten(),
                                                  self.MINs[0], self.MAXs[0],
                                                  periodic = True,
                                                  )
        elif self.n_CVs == 2:
            self._table = mm.Continuous2DFunction(self.list_n_bins[0], self.list_n_bins[1],
                                                  self._bias.flatten(), 
                                                  self.MINs[0], self.MAXs[0], self.MINs[1], self.MAXs[1],
                                                  periodic = True,
                                                  )
        else: print('!')
        self._force.updateParametersInContext(context)
    '''

"""