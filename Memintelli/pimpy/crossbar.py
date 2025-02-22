'''
@author: Yang Ling
2023/11/08
surpporting files for the Pimpy1.2 version
'''
import numpy as np 
from scipy import linalg 

def WordLine(R, Rw, Vins):
    
    if (type(Rw)==int) or (type(Rw)==float):
        Rw = Rw*np.ones(len(R))
        
    assert len(R)==(len(Rw))
    
    R = np.hstack((R, np.zeros(1)))
    Vins = np.hstack((Vins, np.zeros(1)))
    
    Vc = np.zeros((len(R)-2, len(R)))    # Coefficient
    b = -(Vins[1:-1]/R[1:-1])
    
    Rx = np.zeros_like(Rw) 
    Rx[1:-1] = Rw[1:-1] 

    Rx[0] = Rw[0]+R[0] 
    Rx[-1] = Rw[-1]+R[-1] 
    
    for i in range(len(R)-2): 
        
        Vc[i, i] = 1/Rx[i] 
        Vc[i, i+1] = -(1/Rx[i] + 1/R[i+1] + 1/Rx[i+1])
        Vc[i, i+2] = 1/Rx[i+1] 
        
    A = Vc[:,1:-1]

    b = b - Vc[:, 0]*Vins[0] - Vc[:, -1]*Vins[-1]
    
    x = linalg.solve(A, b)

    results_x = np.zeros(len(Vins)-1)
    results_x[1:] = x
    results_x[0] = x[0] + (Vins[0]-x[0]) * Rw[0]/Rx[0] 

    return results_x

def BitLine(R, Rw, Vbit_in, Vwls):
    
    Vins = Vwls
    
    if (type(Rw)==int) or (type(Rw)==float):
        Rw = Rw*np.ones(len(R))
        
    assert len(R)==(len(Rw))
    
    R = np.hstack((np.zeros(1), R))     # Boundary conditions
    Vins = np.hstack((Vbit_in, Vins))
    
    Vc = np.zeros((len(R)-2, len(R)))    # Boundary conditions Coefficient 
    b = -(Vins[1:-1]/R[1:-1])
    
    Rx = np.zeros_like(Rw)
    Rx[1:-1] = Rw[1:-1]

    Rx[0] = Rw[0]+R[0] 
    Rx[-1] = Rw[-1]+R[-1] 
    
    for i in range(len(R)-2): 
        
        Vc[i, i] = 1/Rx[i] 
        Vc[i, i+1] = -(1/Rx[i] + 1/R[i+1] + 1/Rx[i+1]) 
        Vc[i, i+2] = 1/Rx[i+1] 
        
    A = Vc[:,1:-1]                  # the voltage of the input node is known
    b = b - Vc[:, 0]*Vins[0] - Vc[:, -1]*Vins[-1]
    
    x = linalg.solve(A, b) 

    results_x = np.zeros(len(Vins)-1)
    results_x[:-1] = x

    results_x[-1] = x[-1]+(Vins[-1]-x[-1])*Rw[-1]/Rx[-1]
    
    return results_x

def VoltageInput(input, gmatrix, Rw, max_iteration = 20, max_error=1e-5):
    
    vtop = np.zeros((gmatrix.shape[0], gmatrix.shape[1])) 
    for i in range(gmatrix.shape[0]): 
        vtop[i] = input 
    vbot = np.zeros((gmatrix.shape[0], gmatrix.shape[1])) 
    
    if gmatrix.shape[0]==1: 
        vbot[0,:] = WordLine(gmatrix[0,:], Rw, vtop[0,:]) 
        
    else:

        for i in range(max_iteration):
            pre_vtop = vtop + 1e-8
        
            for t in range(gmatrix.shape[0]):
                vbot[t,:] = WordLine(gmatrix[t,:], Rw, vtop[t,:])
        
            for b in range(gmatrix.shape[1]): 
                vtop[:,b] = BitLine(gmatrix[:,b], Rw, input[b], vbot[:, b]) 
            
            if np.sum(np.abs((pre_vtop-vtop)/(pre_vtop)))<max_error:
                break
        
    return vbot[:,-1]/Rw 

def hdot(v_inputs, gmatrix, Rw, max_iteration = 20, max_error=1e-5): 
    
    if len(v_inputs.shape)==1:
        v_inputs = v_inputs.reshape((1, len(v_inputs)))

    if len(gmatrix.shape)==1:
        gmatrix = (gmatrix.reshape((1, len(gmatrix))))
        
    currents = np.zeros((v_inputs.shape[0], gmatrix.shape[1])) 
    
    for i in range(v_inputs.shape[0]):
        currents[i] = VoltageInput(v_inputs[i], gmatrix.T, Rw, max_iteration = max_iteration, max_error=max_error)
        
    return currents