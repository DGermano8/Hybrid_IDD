import numpy as np
import time
import matplotlib.pyplot as plt
from JSF_Solver_BasePython import JumpSwithFlowSimulator
# from JSF_Solver import JumpSwithFlowSimulator

import cProfile

#
#           --------------------------------------------------
#           |                   mWane*R                      |
#           v                                                |
# mBirth*N -----   mBeta*I*S/N     -----      mGamma*I     -----
#     ---> | S |       --->        | I |       --->        | R |
#          -----                   -----                   -----
#          | mDeath*S              | mDeath*I              | mDeath*R
#          V                       V                       V
#


# np.random.seed(3)

# These define the rates of the system
mBeta = 2/7  # Infect "___" people a week
mGamma = 0.5/7  # infecion for "___" weeks
mDeath = 1/(2*365)  # lifespan
mBirth = mDeath
mWane = 0/(2.0*365)

# These are the initial conditions
N0 = 10**6
I0 = 20
R0 = 0
S0 = N0-I0-R0

# How long to simulate for
tFinal = 1000

# These are solver options
dt = 10**-2
SwitchingThreshold = [10**3, 10**3, 10**3]

# kinetic rate parameters
X0 = [S0, I0, R0]

# reactant stoichiometries
nuReactant = [[1, 1, 0],
               [0, 1, 0],
               [1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [0, 0, 1]]

# product stoichiometries
nuProduct = [[0, 2, 0],
                   [0, 0, 1],
                   [2, 0, 0],
                   [1, 1, 0],
                   [1, 0, 1],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 0, 0]]

# stoichiometric matrix
nu = [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(nuProduct, nuReactant)]


# propensity function
def rates_optimized(X, t):
    """
    This is the rate function for the SIR model with demography
    """
    s = X[0]
    i = X[1]
    r = X[2]
    return [mBeta*(s*i)/(s+i+r),
                     mGamma*i,
                     mBirth*s,
                     mBirth*i,
                     mBirth*r,
                     mDeath*s,
                     mDeath*i,
                     mDeath*r,
                     mWane*r]




# identify which reactions are discrete and which are continuous
# make sure that the shape of DoDisc is (3,1)
DoDisc = [1, 1, 1]

# allow S and I to switch, but force R to be continuous
EnforceDo = [0, 0, 0]

stoich = {'nu': nu, 'DoDisc': DoDisc, 'nuReactant': nuReactant, 'nuProduct': nuProduct}

myOpts = {'EnforceDo': EnforceDo, 'dt': dt, 'SwitchingThreshold': SwitchingThreshold}

# Do a benchmark of the following expression
import timeit

# bar = timeit.timeit('rates(X0, 0)', number=100000, globals=globals())
# plumbus = timeit.timeit("rates_optimized([99980, 20, 0], 0)", number=100000, globals=globals())

# >>> timeit.timeit("np.sum(np.arange(10**6))", number=1000, globals=globals())
# 0.3723146120319143
# >>> timeit.timeit("sum(range(10**6))", number=1000, globals=globals())
# 5.511232746008318


# with cProfile.Profile() as pr:
#     pr.enable()


start_time = time.time()  # Record the start time
X, TauArr = JumpSwithFlowSimulator(X0, rates_optimized, stoich, tFinal, myOpts)
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate the elapsed time

print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # pr.disable()
    # pr.print_stats()
    # pr.dump_stats('flamegraph-profile.prof')

# You can make a pretty plot of the time spent in each function with
# flameprof (https://github.com/baverman/flameprof/) and use the
# following command:
#
# $ flameprof requests.prof > requests.svg
#
# This will create a file called requests.svg that you then need to
# open in a browser.

plt.plot(TauArr, X[0], label='S', marker='.', linestyle='-', color='blue')
plt.plot(TauArr, X[1], label='I', marker='.', linestyle='-', color='red')
plt.plot(TauArr, X[2], label='R', marker='.', linestyle='-', color='green')

plt.xlabel('time')
plt.ylabel('Number of People')
plt.title('SIR with Demography')
plt.grid(True)

plt.legend()  # Display legend
plt.show()

plt.yscale("log")
plt.xscale("log")
plt.plot(X[0],X[1], label='S', marker='.', linestyle='-', color='blue')
plt.show()
