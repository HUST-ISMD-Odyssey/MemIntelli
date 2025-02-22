# memmat
"""
@author: Yang Ling
Created on Wed Mar  2 20:37:23 2022
1st Revised on Tue Sep. 6 14:10:13 2022
2nd Revised on Tue Mar. 23 20:23:06 2023
3rd Revised on Thu Dec. 14 09:01:17 2023

"""

import numpy as np
from MemIntelli.pimpy import crossbar
from matplotlib import pyplot as plt

def ABSE(ytest, ypred):
    return np.sum(np.abs((ytest-ypred)/ytest))/(ytest.shape[0]*ytest.shape[1])

def quant_map(mat, blk=(1, 1, 2, 4)):

    assert blk[0] == 1
    bits = np.sum(blk)

    n_blk = len(blk)

    if np.max(np.abs(mat)) == 0:
        data_int = np.zeros((n_blk, mat.shape[0], mat.shape[1]))
    else:
        matq = np.round(mat / np.max(np.abs(mat)) * (2 ** (bits - 1) - 1))
        matq[np.where(matq < 0)] = 2 ** bits + matq[np.where(matq < 0)]
        matq = matq.astype('int')
        data_int = np.zeros((n_blk, mat.shape[0], mat.shape[1]))

        b = 0
        for i in range(n_blk):
            data_int[i, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - i])) >> b
            b += blk[-1 - i]

    return data_int

class Bitslicedpe:
    
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

    def BitSliceVMM(self, x, mat, xblk=(1,1,2,4), mblk=(1,1,2,4)):     #注意x必须为二维向量
    
        mdata = quant_map(mat, blk=mblk)
        xdata = quant_map(x, blk=xblk)
        
        mbits=np.sum(mblk)
        xbits=np.sum(xblk)
        
        n_mblk = len(mblk)
        n_xblk = len(xblk)
        
        out = np.zeros((mat.shape[0], x.shape[1]))
        wi = 0
        
        for i in range(n_xblk):
            out1 = np.zeros((mat.shape[0], x.shape[1]))
            wj = 0
            
            for j in range(n_mblk): 
                if j==(n_mblk-1): 
                    out1 = out1 - np.dot(xdata[j,:, :], mdata[i ,:, :])*2**wj
                else:
                    out1 = out1 + np.dot(xdata[j,:, :], mdata[i ,:, :])*2**wj
                
                wj += mblk[-1-j]
                
            if i==(n_mblk-1):
                out = out - out1*2**wi
                
            else:
                out = out + out1*2**wi
                
            # print(2**wi)  
            wi += xblk[-1-i] 

        return out*np.max(np.abs(mat))*np.max(np.abs(x))/(2**(mbits-1)-1)/(2**(xbits-1)-1)
    
    
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
        
        # G = self.Num2R(matint, mmax) 
        # Vin = self.Num2V(xint, xmax) 
        
        if wire_factor: 
            
            I = crossbar.hdot(Vin, 1 / G, self.wire_resistance) - crossbar.hdot(Vin, 1 / self.LGS * np.ones_like(G), self.wire_resistance)
            maxV = self.vread*np.ones(Vin.shape[1]) 
            minR = 1/self.HGS * np.ones(G.shape[0]) 
            maxR = 1/self.LGS * np.ones(G.shape[0]) 

            adcRef = crossbar.hdot(maxV, minR.reshape(len(minR), 1), self.wire_resistance) - crossbar.hdot(maxV, minR.reshape(len(maxR), 1), self.wire_resistance)

        else:
            # I = np.dot(Vin, G)            # 不减基准值，对于矩阵矢量乘法来说会存在一定误差，但是不影响数据大小关系 
            I = np.dot(Vin, G - self.LGS)      # 使用了补码之后全都是正数，但是依然需要减去基准值 
            
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[1]   # ADC的基准电压，为减小计算量这里省略了TIA，因为它只是缩放，并不会引入任何量化操作
        
        Iq = np.round(I/adcRef * (self.radc-1)) / (self.radc-1) 
        QG = (self.HGS - self.LGS) / (self.g_level-1) 
        
        Num = np.round(Iq / QG / self.vread / (self.g_level-1) * xmax * mmax * adcRef)    # INT运算本身均为整数，因此结果也是整数，对每一次的结果取整，能大幅提升计算的精度
        
        return Num 
    
    def __dot(self, x, mat, xblk=(1,1,2,4), mblk=(1,1,2,4), wire_factor=False):
        
        xint = quant_map(x, blk=xblk)
        matint = quant_map(mat, blk=mblk)
        
        xbits = np.sum(xblk) 
        mbits = np.sum(mblk) 
        
        n_mblk = len(mblk)
        n_xblk = len(xblk)
    
        out = np.zeros((x.shape[0], mat.shape[1])) 
        wi = 0 
        
        for i in range(n_mblk):
            G = self.Num2R(matint[i,:, :], 2**mblk[-1-i]-1)
            out1 = np.zeros((x.shape[0], mat.shape[1])) 
            wj = 0 
            
            for j in range(n_xblk): 
                
                Vin = self.Num2V(xint[j ,:, :], 2**xblk[-1-j]-1)
                
                if j==(n_xblk-1): 
                    out1 = out1 - self.__dot_singleblk(Vin, G, 2**xblk[-1-j]-1, 2**mblk[-1-i]-1, wire_factor) *2**wj
                else: 
                    out1 = out1 + self.__dot_singleblk(Vin, G, 2**xblk[-1-j]-1, 2**mblk[-1-i]-1, wire_factor) *2**wj
                # print(2**wj)
                wj += xblk[-1-j] 
                
            if i==(n_mblk-1):
                out = out - out1*2**wi
            else:
                out = out + out1*2**wi
                
            wi += mblk[-1-i]

        return out*np.max(np.abs(x))*np.max(np.abs(mat))/(2**(xbits-1)-1)/(2**(mbits-1)-1)
    
        
    def MapReduceDot(self, x, mat, xblk=(1,1,2,4), mblk=(1,1,2,4), wire_factor=False):
        
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
                block_out_row += self.__dot(operand_x, operand_m, xblk, mblk, wire_factor) 
                # block_out_row += np.dot(operand_x, operand_m)
                
            result[:, i*self.array_size[1] : (i+1)*self.array_size[1]] = block_out_row 
            
        return result[:n_row, :n_col]

##******************************************** test bench ***************************************#
if __name__ == '__main__':
    # np.random.seed(42)
    a = np.arange(2400).reshape(60, 40)
    # np.random.seed(1)
    b = np.arange(2400).reshape(40, 60)
    c = np.dot(a, b)
    # lgth_fraction = 8
    # c = np.dot(x, b)
    dpe = Bitslicedpe(var=0.0)
    # ch = dpe.dot(x,b, xblk=[4,4,4,4,2,1,1,1,1,1], mblk=[4,4,4,4,2,1,1,1,1,1], wire_factor=False)
    # ch = dpe.dot(x,b, xblk=[1 for i in range(lgth_fraction-4)]+[4], mblk=[1 for i in range(lgth_fraction-4)]+[4], wire_factor=False)
    # ch = dpe.dot(x,b, xblk=[1 for i in range(lgth_fraction)], mblk=[1 for i in range(lgth_fraction)], wire_factor=False)
    # ch = dpe.MapReduceDot(x,b, xblk=[1 for i in range(lgth_fraction-4)]+[4], mblk=[1 for i in range(lgth_fraction-4)]+[4], wire_factor=False)
    ch = dpe.MapReduceDot(a,b, [1,1,2,4,4,4], [1,1,2,4])
    # ch = dpe.BitSliceVMM(x,b)
    # ch = dpe.BitSliceVMM(x,b, xblk=[1 for i in range(lgth_fraction-4)]+[4], mblk=[1 for i in range(lgth_fraction-4)]+[4])

    # sns.heatmap(np.abs(c-ch), cmap='coolwarm')
    errorh = ABSE(c, ch) 
    # errorm = ABSE(c, cm) 
    plt.scatter(c.reshape(-1), ch.reshape(-1)) 
    plt.xlabel('Expected Value of Dot Product')
    plt.ylabel('Measured Value of Dot Product')
    # errors = ABSE(c, cm) 
    print(errorh) 
    plt.show()