# memmatfp
"""
@author: Yang Ling
Created on Wed Mar  2 20:37:23 2022
1st Revised on Tue Sep. 6 20:17:09 2022
2nd Revised on Tue Mar. 23 21:18:19 2023
3rd Revised on Thu Dec. 14 09:01:17 2023

""" 

import numpy as np
from MemIntelli import ABSE, dec_2FP_map, crossbar

class fpmemdpe:
    def __init__(
        self, 
        HGS=1e-5, LGS=1e-7, g_level=16, var=0.1, vnoise = 0.05, wire_resistance=2.93, 
        rdac=256, radc=1024, vread=0.1, array_size=(32, 32)): 
        
        self.HGS = HGS 
        self.LGS = LGS 
        self.g_level = g_level 
        self.var = var 
        self.vnoise = vnoise
        self.wire_resistance = wire_resistance 
        self.rdac = rdac 
        self.radc = radc 
        self.vread = vread 
        self.array_size = array_size 
        
        assert self.rdac>=2
        assert self.radc>=2
        
        # super().__init__(HGS, LGS, g_level, var, wire_resistance, rdac, radc, vread, array_size)

    
    def fpvmm(self, x, mat, xblk=[1,2,2,2,4,4,4,4], mblk=[1,2,2,2,4,4,4,4], bw_e=8): 
        
        Ex, fpx = dec_2FP_map(x, blk=xblk, bw_e=bw_e)
        Em, fpm = dec_2FP_map(mat, blk=mblk, bw_e=bw_e)
        
        nxblk = [1, 1]+xblk
        nmblk = [1, 1]+mblk
        
        out = np.zeros((x.shape[0], mat.shape[1]))
        wi = 0
        for i in range(len(nxblk)):
            wi += nxblk[i]
            
            out1 = np.zeros((x.shape[0], mat.shape[1])) 
            wj = 0 
            for j in range(len(nmblk)): 
                wj += nmblk[j] 
                if j==0: 
                    out1 = out1 - np.dot(fpx[:,:,i], fpm[:, :, j])*(2**(2-wj)) 
                else: 
                    out1 = out1 + np.dot(fpx[:,:,i], fpm[:, :, j])*(2**(2-wj))  
                    
            if i==0: 
                out = out - out1*(2**(2-wi)) 
            else: 
                out = out + out1*(2**(2-wi)) 
                
        return out*(2.**(Ex[0]+Em[0])) 
        

    def Num2V(self, xint, xmax): 
        
        vout = self.vread * np.round(xint/xmax * (self.rdac-1))/(self.rdac-1) 
        
        return vout
    
    def Num2R(self, matint, mmax):
        
        Q_G = (self.HGS-self.LGS)/(self.g_level-1) 
        mat_gq = np.round(matint*(self.g_level-1)/mmax) 
        G = mat_gq * Q_G + self.LGS 
        
        r = np.random.lognormal(0, self.var, size=matint.shape) 
        G = G * r
        
        return G
    
    def __dot_singleblk(self, Vin, G, xmax, mmax, wire_factor=False): 
        
        if wire_factor: 
            
            I = crossbar.hdot(Vin, 1 / G, self.wire_resistance) - crossbar.hdot(Vin, 1 / self.LGS * np.ones_like(G), self.wire_resistance)
            maxV = self.vread * np.ones(Vin.shape[1]) 
            minR = 1/self.HGS * np.ones(G.shape[0]) 
            maxR = 1/self.LGS * np.ones(G.shape[0]) 

            adcRef = crossbar.hdot(maxV, minR.reshape(len(minR), 1), self.wire_resistance) - crossbar.hdot(maxV, minR.reshape(len(maxR), 1), self.wire_resistance)

        else:
            # I = np.dot(Vin, G)             # 不减基准值，对于矩阵矢量乘法来说会存在一定误差，但是不影响数据大小关系 
            I = np.dot(Vin, G - self.LGS)       # 使用了补码之后全都是正数，但是依然需要减去基准值 
            
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[1]   
        
        Iq = np.round(I/adcRef * (self.radc-1)) / (self.radc-1) 
        QG = (self.HGS - self.LGS) / (self.g_level-1) 
            
        Num = np.round(Iq / QG / self.vread / (self.g_level-1) * xmax * mmax * adcRef) 
            
        return Num 
    
    def __dot(self, x, mat, xblk=[1,2,2,2,4,4,4,4], mblk=[1,2,2,2,4,4,4,4], bw_e=8, wire_factor=False):
        
        Ea, xint = dec_2FP_map(x, blk=xblk, bw_e=bw_e)
        Eb, matint = dec_2FP_map(mat, blk=xblk, bw_e=bw_e)
        
        nxblk = [1, 1]+xblk
        nmblk = [1, 1]+mblk
        
        num_xblk = len(nxblk) 
        num_mblk = len(nmblk) 
    
        out = np.zeros((x.shape[0], mat.shape[1])) 
        wi = 0 
        for i in range(num_mblk): 
            
            wi += nmblk[i]
            G = self.Num2R(matint[:,:, i], 2**nmblk[i]-1) 
            out1 = np.zeros((x.shape[0], mat.shape[1])) 
            wj = 0 
            
            for j in range(num_xblk): 
                
                wj += nxblk[j] 
                Vin = self.Num2V(xint[: ,:, j], 2**nxblk[j]-1) 
                
                if j==0: 
                    out1 = out1 - self.__dot_singleblk(Vin, G, 2**nxblk[j]-1, 2**nmblk[i]-1, wire_factor) * 2**(2-wj)
                else: 
                    out1 = out1 + self.__dot_singleblk(Vin, G, 2**nxblk[j]-1, 2**nmblk[i]-1, wire_factor) * 2**(2-wj)
                
            if i==0: 
                out = out - out1* 2**(2-wi) 
            else: 
                out = out + out1* 2**(2-wi) 
                
        # return out*np.max(np.abs(x))*np.max(np.abs(mat))/(2**(xbits-1)-1)/(2**(mbits-1)-1) 
        # return out 
        
        return out * (2.**(Ea[0]+Eb[0])) 
    
    def MapReduceDot(self, x, mat, xblk=[1,2,2,2,4,4,4,4], mblk=[1,2,2,2,4,4,4,4], bw_e=8, wire_factor=False): 
        
        if len(x.shape)==1: 
            x = x.reshape((1, len(x))) 
            
        if len(mat.shape)==1: 
            mat = mat.reshape((1, len(mat))).T 
            
        n_row = x.shape[0] 
        n_col = mat.shape[1] 
        
        polish0 = mat.shape[0] % self.array_size[0] 
        polish1 = mat.shape[1] % self.array_size[1] 
        
        if polish0 != 0: 
            mat = np.hstack((mat, np.zeros((mat.shape[0], self.array_size[1] - polish1))))
            
        if polish1 != 0:
            mat = np.vstack((mat, np.zeros((self.array_size[0] - polish0, mat.shape[1]))))
            x = np.hstack((x, np.zeros((x.shape[0], self.array_size[0] - polish0))))
        
        result = np.zeros((x.shape[0], mat.shape[1])) 
        for i in range(int(mat.shape[1]/self.array_size[1])): 
            block_out_row = 0 
            for j in range(int(mat.shape[0]/self.array_size[0])): 
                operand_x = x[:, j*self.array_size[1] : (j+1)*self.array_size[1]] 
                operand_m = mat[j*self.array_size[0] : (j+1)*self.array_size[0], i*self.array_size[1] : (i+1)*self.array_size[1]] 
                block_out_row += self.__dot(operand_x, operand_m, xblk, mblk, bw_e, wire_factor)
                
                # block_out_row += np.dot(operand_x, operand_m)
                
            result[:, i*self.array_size[1] : (i+1)*self.array_size[1]] = block_out_row 
            
        return result[:n_row, :n_col]


# ##*************************************test bench
if __name__ == '__main__':
    np.random.seed(42) 
    a = np.random.randn(64, 64)  
    np.random.seed(1) 
    b = np.random.randn(64, 64) 
    c = np.dot(a, b) 

    lgth_fraction = 25 
    dpe = fpmemdpe(var=0.05) 
    # ch2 = dpe.MapReduceDot(x,b, xblk=[1 for i in range(lgth_fraction-2)], mblk=[1 for i in range(lgth_fraction-2)], bw_e=8, wire_factor=False)
    cs = dpe.fpvmm(a,b, xblk=[1 for i in range(lgth_fraction-2)], mblk=[1 for i in range(lgth_fraction-2)]) 

    # import seaborn as sns 
    # sns.heatmap(np.abs(c-ch), cmap='coolwarm') 
    # errorh = ABSE(c, ch2) 
    # errorm = ABSE(c, cm) 
    errorh = ABSE(c, cs) 
    import matplotlib.pyplot as plt 
    plt.scatter(c.reshape(-1), cs.reshape(-1)) 
    plt.xlabel('Expected Value of Dot Product')
    plt.ylabel('Measured Value of Dot Product')
    plt.show()
    print(errorh) 