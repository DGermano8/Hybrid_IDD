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
    overFlowAllocation = round(2.5 * len(TimeMesh))

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
    
    NewDiscCompartmemt = np.zeros(nCompartments)
    correctInteger = 0
    
    while ContT < tFinal:
        ContT = ContT + dt 
        iters = iters + 1 

        Dtau = dt 
        
        Xprev = X[:, iters-1] 
        NewDiscCompartmemt = np.zeros(nCompartments)

        # identify which compartment is to be modelled with Discrete and continuous dynamics
        if np.count_nonzero(EnforceDo) !=  nCompartments: #len(EnforceDo):
            
            Props = rates(Xprev, AbsT-Dtau)

            NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, _, _, _ = IsDiscrete(Xprev, nu, Props, rates, dt, AbsT, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes, RandTimes, nCompartments)
            # move Euler mesh to ensure the new distcrete compartment is integer
            correctInteger = 0
            
            if np.count_nonzero(NewDoDisc) > np.count_nonzero(DoDisc):
                # this ^ identifies a state has switched to discrete
                # identify which compartment is the switch

                pos_ind = np.where(np.logical_and(NewDoDisc, np.logical_not(DoDisc)))
                pos = pos_ind[0][0]
                # compute propensities

                # Props = rates(Xprev, AbsT-Dtau)

                # Perform the Forward Euler Step
                dXdt = np.sum(Props * (contCompartment * nu), axis=0).T

                Dtau = np.min([dt,abs((round(Xprev[pos]) - Xprev[pos]) / dXdt[pos])])

                if Dtau < dt:
                    NewDiscCompartmemt = (np.logical_and(NewDoDisc, np.logical_not(DoDisc)) == 1)
                    correctInteger = 1
                    ContT = ContT - dt + Dtau
                    AbsT = AbsT - dt + Dtau
                    Props = rates(Xprev, AbsT-Dtau)

            else:
                contCompartment = NewcontCompartment
                discCompartment = NewdiscCompartment
                DoCont = NewDoCont
                DoDisc = NewDoDisc
        else:
            Props = rates(Xprev, AbsT-Dtau)


        # compute propensities
        # Props = rates(Xprev, AbsT-Dtau)
        # Perform the Forward Euler Step
        dXdt = np.sum(Props * (contCompartment * nu), axis=0).reshape(nCompartments, 1)
        
        Xcurr = X[:, iters-1] + Dtau * (dXdt * DoCont).flatten()
        # TauArr[iters] = ContT

        OriginalDoCont = DoCont.copy()
        if correctInteger:
            contCompartment = NewcontCompartment.copy()
            discCompartment = NewdiscCompartment.copy()
            DoCont = NewDoCont.copy()
            DoDisc = NewDoDisc.copy()
            correctInteger = 0

        # Dont bother doing anything discrete if its all continuous
        stayWhile = (True) * (np.count_nonzero(DoCont) != nCompartments) #len(DoCont))

        TimePassed = 0
        # Perform the Stochastic Loop
        while stayWhile:

            Props = rates(Xprev, AbsT-Dtau)

            # Integrate the cumulative wait times using trapezoid method
            TrapStep = Dtau*0.5*(Props + rates(Xcurr, AbsT))
            sumTimes = sumTimes + TrapStep

            for ii in range(nCompartments):
                if NewDiscCompartmemt[ii] and not EnforceDo[ii]:
                    for jj in range(nRates):
                        if compartInNu[jj, ii]:
                            discCompartment[jj] = 1
                            sumTimes[jj] = 0.0
                            RandTimes[jj] = np.random.rand()

            # identify which events have occurred
            IdEventsOccurred = (RandTimes < (1 - np.exp(-sumTimes))) * discCompartment

            if np.count_nonzero(IdEventsOccurred) > 0:
                tauArray = np.zeros(nRates)
                for kk in range(nRates):

                    if IdEventsOccurred[kk]:
                        # calculate time tau until event using linearisation of integral:
                        # u_k = 1-exp(- integral_{ti}^{t} f_k(s)ds )
                        ExpInt = np.exp(-(sumTimes[kk] - TrapStep[kk]))

                        # Props = rates(Xprev, AbsT-Dtau)
                        # were doing a linear approximation to solve this, so
                        # it may be off, in which case we just fix it to a
                        # small order here
                        Integral = np.log((1 - RandTimes[kk]) / ExpInt)

                        tau_val_1 = Integral/(-1 * Props[kk])

                        # Try Newtons Method to find the time to more accuracy
                        # Error = 1 #This ensures we do atleast one iteration 
                        # howManyHere = 1
                        # while np.any(np.abs(Error) < 10**(-10)) or howManyHere > 10:
                        #     howManyHere += 1
                        #     Props2 = rates(Xprev + ((tau_val_1*(OriginalDoCont))*dXdt).flatten(), AbsT + tau_val_1)
                        #     Error = 0.5 * tau_val_1 * (Props2[kk] + Props[kk]) - Integral
                        #     tau_val_1 = tau_val_1 - 1.0 / Props2[kk] * Error
                            

                        # print(tau_val_1)
                        tau_val_2 = -1
                        tau_val = tau_val_1
                        # if tau_val_1 < 0:
                        #     tau_val_1 = np.abs(tau_val_1)
                        #     if abs(tau_val_1) < dt**(2):
                        #         tau_val_2 = np.abs(tau_val_1)
                        #     tau_val_1 = 0
                        #     tau_val = max(tau_val_1,tau_val_2)
                        #     Dtau = 0.5*Dtau
                        #     sumTimes = sumTimes - TrapStep

                        tauArray[kk] = tau_val
                # identify which reaction occurs first
                if np.sum(tauArray) > 0:
                    tauArray[tauArray==0.0] = np.inf
                    # Identify first event occurrence time and type of event
                    Dtau1 = np.min(tauArray)
                    pos = np.argmin(tauArray)

                    TimePassed = TimePassed + Dtau1
                    AbsT = AbsT + Dtau1
                    
                    # implement first reaction
                    
                    X[:, iters] = X[:, iters-1] + nu[pos, :]
                    Xprev = X[:, iters-1]
                    Xcurr = X[:, iters]
                    TauArr[iters] = AbsT
                    iters = iters + 1

                    Props = rates(Xprev + ((Dtau1*(OriginalDoCont))*dXdt).flatten(), AbsT)

                    # Bring compartments up to date
                    sumTimes = sumTimes - TrapStep
                    TrapStep = Dtau1 * 0.5 * (rates(Xprev, AbsT-Dtau1) + Props)
                    sumTimes = sumTimes + TrapStep

                    # reset timers and sums
                    RandTimes[pos] = np.random.rand()
                    sumTimes[pos] = 0.0

                    # Check if a compartment has become continuous. If so, update the system to this point and
                    # move the FE mesh to this point and 
                    NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, _, _, _ = IsDiscrete(Xprev, nu, Props, rates, Dtau, AbsT-Dtau, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes, RandTimes, nCompartments)
                    if np.count_nonzero(NewDoCont) > np.count_nonzero(DoCont):
                        ContT = ContT - (Dtau-Dtau1)
                        stayWhile = False
                    else:
                        # execute remainder of Euler Step
                        Dtau = Dtau-Dtau1

                    # execute remainder of Euler Step
                    Dtau = Dtau - Dtau1

                else:
                    stayWhile = False
            else:
                stayWhile = False

            if (AbsT > ContT) or (TimePassed >= dt):
                stayWhile = False

        X[:, iters] = Xcurr
        TauArr[iters] = ContT

        if np.count_nonzero(NewDiscCompartmemt) == 1:
            pos = np.argmax(NewDiscCompartmemt)
            X[pos, iters] = round(X[pos, iters]) # type: ignore

            for jj in range(nRates):
                if compartInNu[jj, pos]:
                    discCompartment[jj] = 1
                    sumTimes[jj] = 0.0
                    RandTimes[jj] = np.random.rand()
            NewDiscCompartmemt[pos] = 0

        AbsT = ContT

    return X, TauArr

def IsDiscrete(X, nu, Props, rates, dt, AbsT, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes, RandTimes, nCompartments):
    
    Xprev = X.copy()
    DoDiscTmp = DoDisc.copy()
    DoContTmp = DoCont.copy()    

    # # Calculate dX_ii using vectorized operations
    # dX_ii = dt * np.sum(np.abs(nu) * Props, axis=0)

    # # Calculate conditions for DoContTmp and DoDiscTmp using vectorized operations
    # condition1 = (dX_ii >= SwitchingThreshold[0])
    # condition2 = (X > SwitchingThreshold[1])

    # # Set values for DoContTmp and DoDiscTmp using vectorized operations
    # DoContTmp = np.where(condition1 | condition2, 1, 0).reshape(DoContTmp.shape)
    # DoDiscTmp = np.where(~condition1 & (~condition2), 1, 0).reshape(DoDiscTmp.shape)

    # condition2 = (X > SwitchingThreshold[1])

    # # Set values for DoContTmp and DoDiscTmp using vectorized operations
    # DoContTmp = np.where(condition2, 1, 0).reshape(DoContTmp.shape)
    # DoDiscTmp = np.where((~condition2), 1, 0).reshape(DoDiscTmp.shape)
    # DoDiscTmp = ((X.reshape(DoContTmp.shape) < SwitchingThreshold[1])*(EnforceDo==0)).reshape(DoContTmp.shape)
    DoDiscTmp = (X.reshape(DoDisc.shape) < SwitchingThreshold[1]) * (EnforceDo == 0)
    DoContTmp = np.ones(DoDiscTmp.shape) - DoDiscTmp

    # are_equal = DoDiscTmp.flatten() == DoDisc.flatten()
    are_equal = DoDiscTmp == DoDisc
    if (np.count_nonzero(are_equal) == nCompartments): #len(are_equal)):
        discCompartmentTmp = discCompartment.copy()
        contCompartmentTmp = contCompartment.copy()
    else:
        discCompartmentTmp = np.zeros(discCompartment.shape)
        contCompartmentTmp = np.ones(contCompartment.shape)

        for idx, x in enumerate(X):
            for compartIdx in range(compartInNu.shape[0]):
                if  EnforceDo[idx]==0:
                    if DoDiscTmp[idx]==1 and compartInNu[compartIdx, idx]==1:
                        discCompartmentTmp[compartIdx] = 1
                else:
                    if DoDisc[idx]==1 and compartInNu[compartIdx, idx]==1:
                        discCompartmentTmp[compartIdx] = 1
        contCompartmentTmp -= discCompartmentTmp

    return DoDiscTmp, DoContTmp, discCompartmentTmp, contCompartmentTmp, sumTimes, RandTimes, Xprev