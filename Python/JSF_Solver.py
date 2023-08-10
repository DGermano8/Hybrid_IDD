import numpy as np

def JumpSwithFlowSimulator(x0, rates, stoich, times, options):
    # predefine and initialise the system
    # TODO - add default options

    X0 = x0
    nu = stoich["nu"]
    nuReactant = stoich["nuReactant"]
    DoDisc = stoich["DoDisc"]
    DoCont = np.ones(DoDisc.shape)-DoDisc

    tFinal = times[-1]
    dt = options["dt"]
    EnforceDo = options["EnforceDo"]
    SwitchingThreshold = options["SwitchingThreshold"]

    nRates, nCompartments = nu.shape

    # identify which compartment is in which reaction:
    # LogicalCompartInNu = (np.logical_or(np.logical_not(nu), np.logical_not(nuReactant)))
    # compartInNu = LogicalCompartInNu.astype(int)

    NuComp = (nu != 0)
    ReactComp = (nuReactant != 0)
    compartInNu = ( (NuComp+ReactComp) != 0)

    discCompartment = np.dot(compartInNu, DoDisc)
    discCompartment = discCompartment
    contCompartment = np.ones(discCompartment.shape)-discCompartment

    # initialise discrete sum compartments
    integralOfFiringTimes = np.zeros(nRates).reshape(nRates, 1)
    randTimes = np.random.rand(nRates).reshape(nRates, 1)
    # print(RandTimes)
    tauArray = np.zeros(nRates).reshape(nRates, 1)

    TimeMesh = np.arange(0, tFinal+dt, dt)
    overFlowAllocation = round(10 * len(TimeMesh))

    # initialise solution arrays
    X = np.zeros((nCompartments, overFlowAllocation))
    X[:, 0] = X0
    TauArr = np.zeros(overFlowAllocation)
    iters = 0

    # Track Absolute time
    AbsT = 0
    ContT = 0

    Xprev = X0
    Xcurr = np.zeros(nCompartments)
    
    NewDiscCompartmemt = np.zeros(nCompartments, dtype=int)
    correctInteger = 0

   
    while ContT < tFinal:

        Xprev = X[:,iters]
        Dtau = dt
        Props = rates(Xprev, ContT)

        # check if any states change in this step
        Dtau, correctInteger, DoDisc, DoCont, discCompartment, contCompartment, NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, NewDiscCompartmemt = UpdateCompartmentRegime(dt, Xprev, Dtau, Props, nu, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments)

        # Perform the Forward Euler Step
        dXdt = ComputedXdt(Xprev, Props, nu, contCompartment, nCompartments)
        
        Xcurr = X[:, iters] + Dtau * (dXdt * DoCont).flatten()

        # Update the discrete compartments, if a state has just become discrete
        OriginalDoCont = DoCont.copy()
        if correctInteger == 1:
            contCompartment = NewcontCompartment.copy()
            discCompartment = NewdiscCompartment.copy()
            DoCont = NewDoCont.copy()
            DoDisc = NewDoDisc.copy()

        TimePassed = 0
        # Perform the Stochastic Loop
        stayWhile = (True) * (np.count_nonzero(DoCont) != nCompartments)
        AbsT = ContT
        while stayWhile:
            
            Props = rates(Xprev, AbsT)
            integralStep = ComputeIntegralOfFiringTimes(Dtau, Props, rates, Xprev, Xcurr, AbsT)
            integralOfFiringTimes = integralOfFiringTimes + integralStep

            # If any of the components have just become discrete, we need to update the integralOfFiringTimes and randTimes
            if correctInteger == 1:
                for ii in range(nCompartments):
                    if NewDiscCompartmemt[ii] and not EnforceDo[ii]:
                        for jj in range(nRates):
                            if compartInNu[jj, ii]:
                                discCompartment[jj] = 1
                                integralOfFiringTimes[jj] = 0.0
                                randTimes[jj] = np.random.rand()
            
            # Identify which reactions have fired
            firedReactions = (0 > randTimes - (1 - np.exp(-integralOfFiringTimes))) * discCompartment
            
            if np.count_nonzero(firedReactions) > 0:
                # Identify which reactions have fired
                tauArray = ComputeFiringTimes(firedReactions,integralOfFiringTimes,randTimes,Props,dt,nRates,integralStep)
                
                if np.sum(tauArray) > 0:
                    
                    # Update the discrete compartments
                    Xcurr, Xprev, integralOfFiringTimes, integralStep, randTimes, TimePassed, AbsT = ImplementFiredReaction(tauArray ,integralOfFiringTimes,randTimes,Props,rates,integralStep,TimePassed, AbsT, X, iters, nu, dXdt, OriginalDoCont)

                    iters = iters + 1
                    X[:, iters] = Xcurr
                    TauArr[iters] = AbsT

                    stayWhile = False

                else:
                    stayWhile = False
            else:
                stayWhile = False

            if TimePassed > Dtau:
                stayWhile = False
        
        indicator = 0 if TimePassed > 0 else 1
        ContT = ContT + Dtau*indicator + TimePassed
        
        
        iters = iters + 1
        X[:, iters] = Xcurr
        TauArr[iters] = ContT
            
        if correctInteger == 1:
            pos = np.argmax(NewDiscCompartmemt)
            X[pos, iters] = round(X[pos, iters]) # type: ignore

            for jj in range(nRates):
                if compartInNu[jj, pos]:
                    discCompartment[jj] = 1
                    integralOfFiringTimes[jj] = 0.0
                    randTimes[jj] = np.random.rand()
            NewDiscCompartmemt[pos] = 0

    # pos = np.argmax(TauArr)
    trimmed_TauArr = TauArr[:iters + 1]
    trimmed_X = X[:, :iters + 1]
    return trimmed_X, trimmed_TauArr

def ComputeFiringTimes(firedReactions,integralOfFiringTimes,randTimes,Props,dt,nRates,integralStep):
    tauArray = np.zeros(nRates)
    for kk in range(nRates):
        if firedReactions[kk]:
             # calculate time tau until event using linearisation of integral:
            ExpInt = np.exp(-(integralOfFiringTimes[kk] - integralStep[kk]))
            Integral = np.log((1 - randTimes[kk]) / ExpInt)
            tau_val_1 = Integral/(-1 * Props[kk])

            tau_val_2 = -1
            tau_val = tau_val_1
            if tau_val_1 < 0:
                tau_val_1 = np.abs(tau_val_1)
                if abs(tau_val_1) < dt**(2):
                    tau_val_2 = np.abs(tau_val_1)
                tau_val_1 = 0
                tau_val = max(tau_val_1,tau_val_2)

            tauArray[kk] = tau_val
            
    return tauArray

def ImplementFiredReaction(tauArray ,integralOfFiringTimes,randTimes,Props,rates,integralStep,TimePassed, AbsT, X, iters, nu, dXdt, OriginalDoCont):
    tauArray[tauArray==0.0] = np.inf
     # Identify first event occurrence time and type of event
    DtauMin = np.min(tauArray)
    pos = np.argmin(tauArray)

    TimePassed = TimePassed + DtauMin
    AbsT = AbsT + DtauMin

    # implement first reaction
    Xcurr = X[:, iters] + nu[pos, :] + DtauMin*(dXdt * OriginalDoCont).flatten()
    Xprev = X[:, iters]

    # update the integralOfFiringTimes and randTimes up to time step DtauMin
    integralOfFiringTimes = integralOfFiringTimes - integralStep
    # Props = rates(Xprev, AbsT)
    integralStep = ComputeIntegralOfFiringTimes(DtauMin, Props, rates, Xprev, Xcurr, AbsT)
    integralOfFiringTimes = integralOfFiringTimes + integralStep

    integralOfFiringTimes[pos] = 0.0
    randTimes[pos] = np.random.rand()

    return Xcurr, Xprev, integralOfFiringTimes, integralStep, randTimes, TimePassed, AbsT

def ComputeIntegralOfFiringTimes(Dtau, Props, rates, Xprev, Xcurr, AbsT):
    Props = rates(Xprev, AbsT)
    # Integrate the cumulative wait times using trapezoid method
    integralStep = Dtau*0.5*(Props + rates(Xcurr, AbsT+Dtau))
    return integralStep

def ComputedXdt(Xprev, Props, nu, contCompartment, nCompartments):
    dXdt = np.sum(Props * (contCompartment * nu), axis=0).reshape(nCompartments, 1)
    return dXdt

def UpdateCompartmentRegime(dt, Xprev, Dtau, Props, nu, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments):
    # check if any states change in this step
    NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment = IsDiscrete(Xprev, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments)
    correctInteger = 0
    NewDiscCompartmemt = np.zeros(nCompartments, dtype=int)
    # identify if a state has just become discrete
    if np.count_nonzero(NewDoDisc) > np.count_nonzero(DoDisc):
        # identify which compartment has just switched
        pos_ind = np.where(np.logical_and(NewDoDisc, np.logical_not(DoDisc)))
        pos = pos_ind[0][0]

        # Here, we identify the time to the next integer
        dXdt = np.sum(Props * (contCompartment * nu), axis=0).T
        Dtau = np.min([dt,abs((round(Xprev[pos]) - Xprev[pos]) / dXdt[pos])])

        # If the time to the next integer is less than the time step, we need to move the mesh
        if Dtau < dt:
            NewDiscCompartmemt = (np.logical_and(NewDoDisc, np.logical_not(DoDisc)) == 1).astype(int)
            correctInteger = 1
    else:
        contCompartment = NewcontCompartment
        discCompartment = NewdiscCompartment
        DoCont = NewDoCont
        DoDisc = NewDoDisc

    return Dtau, correctInteger, DoDisc, DoCont, discCompartment, contCompartment, NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, NewDiscCompartmemt

def IsDiscrete(X, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments): 
    # Check if any compartments should be switched to continuous
    DoDiscTmp = (X.reshape(DoDisc.shape) < SwitchingThreshold[1]).astype(int)
    # DoContTmp = np.ones(DoDiscTmp.shape) - DoDiscTmp
    DoContTmp = DoDiscTmp^1

    # Set values for DoContTmp and DoDiscTmp using original values
    for idx, x in enumerate(EnforceDo):
        if x == 1:
            DoDiscTmp[idx] = DoDisc[idx]
            DoContTmp[idx] = DoCont[idx]            

    # Check if the new DoDiscTmp and DoContTmp are the same as the old ones
    are_equal = DoDiscTmp == DoDisc
    if (np.count_nonzero(are_equal) == nCompartments):
        discCompartmentTmp = discCompartment.copy()
        contCompartmentTmp = contCompartment.copy()
    else:
        # If not, update the compartments
        discCompartmentTmp = np.zeros(discCompartment.shape, dtype=int)

        for idx, x in enumerate(X):
            for compartIdx in range(compartInNu.shape[0]):
                if  EnforceDo[idx]==0:
                    if DoDiscTmp[idx]==1 and compartInNu[compartIdx, idx]==1:
                        discCompartmentTmp[compartIdx] = 1
                else:
                    if DoDisc[idx]==1 and compartInNu[compartIdx, idx]==1:
                        discCompartmentTmp[compartIdx] = 1
        # contCompartmentTmp = np.ones(contCompartment.shape) - discCompartmentTmp
        contCompartmentTmp = discCompartmentTmp^1

    return DoDiscTmp, DoContTmp, discCompartmentTmp, contCompartmentTmp