import numpy as np
import math

def JumpSwithFlowSimulator2(x0, rates, stoich, times, options):
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
    contCompartment = np.ones(nCompartments)-discCompartment

    # initialise discrete sum compartments
    integralOfFiringTimes = np.zeros((nRates,1), dtype=float).flatten()
    # randTimes = np.random.rand(nRates).reshape(nRates, 1)
    randTimes = np.random.rand(nRates,1).flatten()

    tauArray = np.zeros((nRates,1), dtype=float)

    TimeMesh = np.arange(0, tFinal+dt, dt)
    overFlowAllocation = round(10 * len(TimeMesh))

    # initialise solution arrays
    X = np.zeros((nCompartments, overFlowAllocation), dtype=float)
    X[:, 0] = X0

    TauArr = np.zeros(overFlowAllocation, dtype=float)
    iters = 0

    # Track Absolute time
    AbsT = 0
    ContT = 0

    Xprev = X0
    Xcurr = np.zeros(nCompartments, dtype=float)
    
    NewDiscCompartmemt = np.zeros(nCompartments, dtype=int)
    correctInteger = 0

    while ContT < tFinal:
        print("ContT: = ",ContT)

        Xprev = X[:,iters]
        Dtau = dt
        Props = rates(Xprev, ContT)

        # check if any states change in this step
        Dtau, correctInteger, DoDisc, DoCont, discCompartment, contCompartment, NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, NewDiscCompartmemt = UpdateCompartmentRegime(dt, Xprev, Dtau, Props, nu, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments,nRates)
        # Perform the Forward Euler Step
        dXdt = ComputedXdt(Xprev, Props, nu, contCompartment, nCompartments)

        # Xcurr = X[:, iters] + Dtau * (dXdt * DoCont).flatten()
        Xcurr = X[:, iters] + Dtau * (dXdt * DoCont)

        # Update the discrete compartments, if a state has just become discrete
        OriginalDoCont = DoCont.copy()
        if correctInteger == 1:
            contCompartment = NewcontCompartment.copy()
            discCompartment = NewdiscCompartment.copy()
            DoCont = NewDoCont.copy()
            DoDisc = NewDoDisc.copy()

        
        # Perform the Stochastic Loop
        stayWhile = (True) * (np.count_nonzero(DoCont) != nCompartments)
        AbsT = ContT
        DtauContStep = Dtau
        TimePassed = 0
        
        while stayWhile:
            
            if TimePassed > 0:
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
            firedReactions = (0 > randTimes - (1 - np.exp(-integralOfFiringTimes.T))) * discCompartment.flatten()

            if np.count_nonzero(firedReactions) > 0:
                # Identify which reactions have fired
                tauArray = ComputeFiringTimes(firedReactions,integralOfFiringTimes,randTimes,Props,dt,nRates,integralStep)
                
                if np.count_nonzero(tauArray) > 0:
                    
                    # Update the discrete compartments
                    Xcurr, Xprev, integralOfFiringTimes, integralStep, randTimes, TimePassed, AbsT, DtauMin = ImplementFiredReaction(tauArray ,integralOfFiringTimes,randTimes,Props,rates,integralStep,TimePassed, AbsT, X, iters, nu, dXdt, OriginalDoCont)

                    iters = iters + 1
                    X[:, iters] = Xcurr
                    TauArr[iters] = AbsT

                    Dtau = Dtau - DtauMin

                else:
                    stayWhile = False
            else:
                stayWhile = False

            if TimePassed >= DtauContStep:
                stayWhile = False
        
        # indicator = 0 if TimePassed > 0 else 1
        # ContT = ContT + Dtau*indicator + TimePassed
        # iters = iters + 1
        # X[:, iters] = Xcurr
        # TauArr[iters] = ContT

        iters = iters + 1
        ContT = ContT + DtauContStep
        TauArr[iters] = ContT
        # X[:,iters] = X[:,iters-1] + (DtauContStep-TimePassed)*(dXdt * DoCont).flatten()
        X[:,iters] = X[:,iters-1] + (DtauContStep-TimePassed)*(dXdt * DoCont)

        if correctInteger == 1:
            pos = np.argmax(NewDiscCompartmemt)
            X[pos, iters] = round(X[pos, iters])

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
    tauArray = np.zeros(nRates, dtype=float)
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
    # Xcurr = X[:, iters] + nu[pos, :] + DtauMin*(dXdt * OriginalDoCont).flatten()
    Xcurr = X[:, iters] + nu[pos, :] + DtauMin*(dXdt * OriginalDoCont)
    Xprev = X[:, iters]

    # update the integralOfFiringTimes and randTimes up to time step DtauMin
    integralOfFiringTimes = integralOfFiringTimes - integralStep
    # Props = rates(Xprev, AbsT)
    integralStep = ComputeIntegralOfFiringTimes(DtauMin, Props, rates, Xprev, Xcurr, AbsT)
    integralOfFiringTimes = integralOfFiringTimes + integralStep

    integralOfFiringTimes[pos] = 0.0
    randTimes[pos] = np.random.rand()

    return Xcurr, Xprev, integralOfFiringTimes, integralStep, randTimes, TimePassed, AbsT, DtauMin

def ComputeIntegralOfFiringTimes(Dtau, Props, rates, Xprev, Xcurr, AbsT):
    # Integrate the cumulative wait times using trapezoid method
    integralStep = Dtau*0.5*(np.array(Props) + np.array(rates(Xcurr, AbsT+Dtau)))

    # temp = ArrayPlusAB(Props , rates(Xcurr, AbsT+Dtau))
    # integralStep = (ScalarTimesArray(Dtau*0.5,temp))
    return integralStep

def ComputedXdt(Xprev, Props, nu, contCompartment, nCompartments):
    # dXdt = np.sum(Props * (contCompartment * nu), axis=0).reshape(nCompartments, 1)
    dXdt = np.dot(Props, contCompartment * nu)
    return dXdt

def UpdateCompartmentRegime(dt, Xprev, Dtau, Props, nu, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments,nRates):
    # check if any states change in this step
    NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment = IsDiscrete(Xprev, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments,nRates)
    correctInteger = 0
    NewDiscCompartmemt = np.zeros(nCompartments, dtype=int)
    print("NewDiscCompartmemt: = ",NewDiscCompartmemt)
    # identify if a state has just become discrete
    if np.count_nonzero(NewDoDisc) > np.count_nonzero(DoDisc):
        # identify which compartment has just switched
        pos_ind = np.where(np.logical_and(NewDoDisc, np.logical_not(DoDisc)))
        pos = pos_ind[0][0]

        # Here, we identify the time to the next integer
        # dXdt = np.dot(Props.T, contCompartment * nu).flatten()
        dXdt = np.dot(Props, contCompartment * nu)

        # dXdt = np.dot(Props.T, contCompartment * nu).flatten()


        Dtau = np.min([dt,abs((round(Xprev[pos]) - Xprev[pos]) / dXdt[pos])])

        # If the time to the next integer is less than the time step, we need to move the mesh
        if Dtau < dt:
            print("NewDoDisc: = ",NewDoDisc)
            print("Shape NewDoDisc: = ",NewDoDisc.shape)
            print("DoDisc: = ",DoDisc)
            print("Shape DoDisc: = ",DoDisc.shape)
            # NewDiscCompartmemt = (np.logical_and(NewDoDisc, np.logical_not(DoDisc)) == 1).astype(int)
            # NewDiscCompartmemt = (np.logical_and(NewDoDisc, np.logical_not(DoDisc))).astype(int)
            NewDiscCompartmemt = [a - b for a, b in zip(NewDoDisc, DoDisc)]



            correctInteger = 1
    else:
        contCompartment = NewcontCompartment
        discCompartment = NewdiscCompartment
        DoCont = NewDoCont
        DoDisc = NewDoDisc
    
    print("NewDiscCompartmemt: = ",NewDiscCompartmemt)
    print("NewDiscCompartmemt Shape: = ",NewDiscCompartmemt.shape)

    return Dtau, correctInteger, DoDisc, DoCont, discCompartment, contCompartment, NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, NewDiscCompartmemt

def IsDiscrete(X, SwitchingThreshold, DoDisc, DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, nCompartments, nRates): 
    # Check if any compartments should be switched to continuous
    DoDiscTmp = (X  < SwitchingThreshold).astype(int)
    # DoContTmp = DoDiscTmp^1
    DoContTmp = (np.ones([nCompartments], dtype=int) - DoDiscTmp)


    # Set values for DoContTmp and DoDiscTmp using original values
    for idx, x in enumerate(EnforceDo):
        if x == 1:
            DoDiscTmp[idx] = DoDisc[idx]
            DoContTmp[idx] = DoCont[idx]            

    # Check if the new DoDiscTmp and DoContTmp are the same as the old ones
    are_equal = (DoDiscTmp == DoDisc)
    if (np.count_nonzero(are_equal) == nCompartments):
        discCompartmentTmp = discCompartment.copy()
        contCompartmentTmp = contCompartment.copy()
    else:
        # If not, update the compartments
        discCompartmentTmp = np.zeros([nRates,1], dtype=int)

        for idx in range(nCompartments):
            for compartIdx in range(nRates):
                if  EnforceDo[idx]==0:
                    if DoDiscTmp[idx]==1 and compartInNu[compartIdx, idx]==1:
                        discCompartmentTmp[compartIdx] = 1
                else:
                    if DoDisc[idx]==1 and compartInNu[compartIdx, idx]==1:
                        discCompartmentTmp[compartIdx] = 1
        # contCompartmentTmp = discCompartmentTmp^1
        contCompartmentTmp = np.ones([nRates,1], dtype=int) - discCompartmentTmp


    return DoDiscTmp, DoContTmp, discCompartmentTmp, contCompartmentTmp


# these are helper functions to make it readable
def ArraySubtractAB(ArrayA, ArrayB):
    AMinusB = [a - b for a, b in zip(ArrayA, ArrayB)]
    return AMinusB
def ArrayPlusAB(ArrayA, ArrayB):
    APlusB = [a + b for a, b in zip(ArrayA, ArrayB)]
    return APlusB
def MatrixSubtractAB(MatrixA,MatrixB):
    AMinusB = [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(MatrixA, MatrixB)]
    return AMinusB
def MatrixPlusAB(MatrixA,MatrixB):
    APlusB = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(MatrixA, MatrixB)]
    return APlusB
def ScalarTimesArray(alpha,Array):
    result = [element * alpha for element in Array]
    return result