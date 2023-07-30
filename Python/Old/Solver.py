import numpy as np

def MovingFEMesh_cdsSimulator(x0, rates, stoich, times, options):

    X0 = x0
    nu = stoich["nu"]
    DoDisc = stoich["DoDisc"]
    DoCont = np.ones(DoDisc.shape)-DoDisc

    tFinal = times[-1]
    dt = options["dt"]
    EnforceDo = options["EnforceDo"]
    SwitchingThreshold = options["SwitchingThreshold"]

    nRates, nCompartments = nu.shape

    # identify which compartment is in which reaction:
    compartInNu = nu != 0
    discCompartment = np.dot(compartInNu, DoDisc)
    discCompartment = discCompartment
    contCompartment = np.ones(discCompartment.shape)-discCompartment

    # initialise discrete sum compartments
    sumTimes = np.zeros(nRates).reshape(nRates, 1)
    RandTimes = np.random.rand(nRates).reshape(nRates, 1)
    # print(RandTimes)
    tauArray = np.zeros(nRates).reshape(nRates, 1)

    TimeMesh = np.arange(0, tFinal+dt, dt)
    overFlowAllocation = round(4 * len(TimeMesh))

    # initialise solution arrays
    X = np.zeros((nCompartments, overFlowAllocation))
    X[:, 0] = X0
    TauArr = np.zeros(overFlowAllocation)
    iters = 0

    # Track Absolute time
    AbsT = dt
    ContT = 0

    Xprev = X0
    Xcurr = np.zeros(nCompartments)
    while ContT < tFinal:
        ContT = ContT + dt
        iters = iters + 1

        Dtau = dt
        Xprev = X[:, iters-1]
        correctInteger = 0

        # identify which compartment is to be modelled with Discrete and continuous dynamics
        if np.sum(EnforceDo) != len(EnforceDo):

            NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, _, _, _ = IsDiscrete(Xprev, nu, rates, dt, AbsT, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes, RandTimes)
            # move Euler mesh to ensure the new distcrete compartment is integer
            correctInteger = 0
            
            if np.count_nonzero(NewDoDisc) > np.count_nonzero(DoDisc):
                # this ^ identifies a state has switched to discrete
                # identify which compartment is the switch
                pos = np.argmax(NewDoDisc - DoDisc)

                # compute propensities
                Props = rates(Xprev, AbsT-Dtau)

                # Perform the Forward Euler Step
                dXdt = np.sum(Props * (contCompartment * nu), axis=0).T

                Dtau = abs((round(Xprev[pos]) - Xprev[pos]) / dXdt[pos])
                if dt < Dtau:
                    # your threshold is off, stay continuous
                    # slow down the FE solver so dont overstep the change
                    Dtau = 0.75 * dt
                else:
                    correctInteger = 1
                    ContT = ContT - dt + Dtau
                    AbsT = AbsT - dt + Dtau

            else:
                contCompartment = NewcontCompartment
                discCompartment = NewdiscCompartment
                DoCont = NewDoCont
                DoDisc = NewDoDisc

        # compute propensities
        Props = rates(Xprev, AbsT-Dtau)
        # Perform the Forward Euler Step
        dXdt = np.sum(Props * (contCompartment * nu), axis=0).reshape(nCompartments, 1)
        
        X[:, iters] = X[:, iters-1] + Dtau * (dXdt * DoCont).flatten()

        TauArr[iters] = ContT

        if correctInteger:
            contCompartment = NewcontCompartment.copy()
            discCompartment = NewdiscCompartment.copy()
            DoCont = NewDoCont.copy()

            for ii in range(len(Xprev)):
                if not EnforceDo[ii]:
                    X[ii, iters] = round(X[ii, iters])
                    for jj in range(compartInNu.shape[0]):
                        if NewDoDisc[ii] and compartInNu[jj, ii]:
                            discCompartment[jj] = 1
                            if not DoDisc[ii]:
                                sumTimes[jj] = 0.0
                                RandTimes[jj] = np.random.rand()
            DoDisc = NewDoDisc.copy()

        # Dont bother doing anything discrete if its all continuous
        stayWhile = (True) * (np.sum(DoCont) != len(DoCont))

        TimePassed = 0
        # Perform the Stochastic Loop
        while stayWhile:

            Xprev = X[:, iters-1]
            Xcurr = X[:, iters]

            # Integrate the cumulative wait times using trapezoid method
            # print("Xprev")
            # print(Xprev)
            # print(Xcurr)
            # print("Rates are:")
            # print(rates(Xprev.T, AbsT-Dtau))
            # print(rates(Xcurr.T, AbsT))
            TrapStep = Dtau*0.5*(rates(Xprev, AbsT-Dtau) + rates(Xcurr, AbsT))
            sumTimes = sumTimes + TrapStep

            # print("sumTimes")
            # print(sumTimes)
            # identify which events have occurred
            IdEventsOccurred = (RandTimes < (1 - np.exp(-sumTimes))) * discCompartment

            if np.sum(IdEventsOccurred) > 0:
                tauArray = np.zeros(nRates)
                for kk in range(len(IdEventsOccurred)):

                    if IdEventsOccurred[kk]:
                        # calculate time tau until event using linearisation of integral:
                        # u_k = 1-exp(- integral_{ti}^{t} f_k(s)ds )
                        ExpInt = np.exp(-(sumTimes[kk] - TrapStep[kk]))

                        Props = rates(Xprev, AbsT-Dtau)
                        # were doing a linear approximation to solve this, so
                        # it may be off, in which case we just fix it to a
                        # small order here
                        tau_val_1 = np.log((1 - RandTimes[kk]) / ExpInt) / (-1 * Props[kk])
                        tau_val_2 = -1
                        tau_val = tau_val_1
                        if tau_val_1 < 0:
                            if abs(tau_val_1) < dt**(2):
                                tau_val_2 = abs(tau_val_1)
                            tau_val_1 = 0
                            tau_val = max(tau_val_1,tau_val_2)
                            Dtau = 0.5*Dtau
                            sumTimes = sumTimes - TrapStep

                        tauArray[kk] = tau_val
                # identify which reaction occurs first
                if np.sum(tauArray) > 0:
                    tauArray[tauArray==0.0] = np.inf
                    # Identify first event occurrence time and type of event
                    Dtau1 = np.min(tauArray)
                    pos = np.argmin(tauArray)

                    TimePassed = TimePassed + Dtau1
                    AbsT = AbsT + Dtau1
                    # print("absT = ")
                    # print(AbsT)
                    
                    # implement first reaction
                    iters = iters + 1
                    X[:, iters] = X[:, iters-1] + nu[pos, :]
                    Xprev = X[:, iters-1]
                    TauArr[iters] = AbsT

                    # Bring compartments up to date
                    sumTimes = sumTimes - TrapStep
                    TrapStep = Dtau1 * 0.5 * (rates(Xprev, AbsT-Dtau1) + rates(Xprev + ((Dtau1*(np.ones(DoDisc.shape)-DoDisc))*dXdt).flatten(), AbsT))
                    sumTimes = sumTimes + TrapStep

                    # reset timers and sums
                    RandTimes[pos] = np.random.rand()
                    sumTimes[pos] = 0.0

                    # execute remainder of Euler Step
                    Dtau = Dtau - Dtau1

                else:
                    stayWhile = False
            else:
                stayWhile = False

            if (AbsT > ContT) or (TimePassed >= dt):
                stayWhile = False
        
        AbsT = ContT

    return X, TauArr


def IsDiscrete(X, nu, rates, dt, AbsT, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes, RandTimes):
    
    Xprev = X.copy()
    DoDiscTmp = DoDisc.copy()
    DoContTmp = DoCont.copy()

    discCompartmentTmp = np.zeros(discCompartment.shape)
    contCompartmentTmp = np.ones(contCompartment.shape)

    for idx, x in enumerate(X):
        if  EnforceDo[idx]==0:
            dX_ii = dt * np.sum(abs(nu[:, idx]) * rates(X, AbsT).T)

            if dX_ii >= SwitchingThreshold[0]:
                DoContTmp[idx] = 1
                DoDiscTmp[idx] = 0
            elif x < SwitchingThreshold[1]:
                DoContTmp[idx] = 0
                DoDiscTmp[idx] = 1
            else:
                DoContTmp[idx] = 0
                DoDiscTmp[idx] = 1

    for idx, x in enumerate(X):
        if  EnforceDo[idx]==0:
            for compartIdx in range(compartInNu.shape[0]):
                if DoDiscTmp[idx]==1 and compartInNu[compartIdx, idx]==1:
                    discCompartmentTmp[compartIdx] = 1
        elif EnforceDo[idx]==1:
            for compartIdx in range(compartInNu.shape[0]):
                if DoDisc[idx]==1 and compartInNu[compartIdx, idx]==1:
                    discCompartmentTmp[compartIdx] = 1

    contCompartmentTmp -= discCompartmentTmp

    return DoDiscTmp, DoContTmp, discCompartmentTmp, contCompartmentTmp, sumTimes, RandTimes, Xprev


# def IsDiscrete(X, nu, rates, dt, AbsT, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes, RandTimes):

#     Xprev = X.copy()
#     DoDiscTmp = DoDisc.copy()
#     DoContTmp = DoCont.copy()

#     discCompartmentTmp = np.zeros(discCompartment.shape)
#     contCompartmentTmp = np.ones(contCompartment.shape)

#     for idx, x in enumerate(X):
#         if not EnforceDo[idx]:
#             dX_ii = dt * np.sum(abs(nu[:, idx]) * rates(X, AbsT))
#             DoDiscTmp[idx], DoContTmp[idx] = (1, 0) if dX_ii >= SwitchingThreshold[0] or x >= SwitchingThreshold[1] else (0, 1)

#             for compartIdx in range(compartInNu.shape[0]):
#                 if DoDiscTmp[idx] and compartInNu[compartIdx, idx]:
#                     discCompartmentTmp[compartIdx] = 1
#         else:
#             for compartIdx in range(compartInNu.shape[0]):
#                 if DoDisc[idx] and compartInNu[compartIdx, idx]:
#                     discCompartmentTmp[compartIdx] = 1

#     contCompartmentTmp -= discCompartmentTmp

#     return DoDiscTmp, DoContTmp, discCompartmentTmp, contCompartmentTmp, sumTimes, RandTimes, Xprev
