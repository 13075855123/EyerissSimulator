import numpy as np
from . import conf
from . import IO2
from . import Pooling
from . import Activiation

class Hive():

    def __init__(self, EyerissF, mode="auto"):
        self.mode = mode
        self.EyerissF = EyerissF
        self.RLE = IO2.RLE(RateNeed = 0)
        self.GLB = conf.GLB
        self.stride = 1  # 🌟 新增：默认步长为 1
        #TODO: read/write communications between GLB, PEArray, individual is missing
        
        #maybe wrap it to a separate class
        self.m = 1
        self.n = 1
        self.p = 1
        self.q = 1
        self.e = 1
        self.r = 1
        self.t = 1
        
    def PreProcess(self, *args):
        if len(args) == 1:
            return self.RLE.Decompress(args[0])
        return [self.RLE.Decompress(arg) for arg in args]

    def PostProcess(self, OfMaps):
        return self.RLE.Compress(OfMaps)
        
    # 🌟 修改：接受 stride 参数
    def Conv2d(self, Pictures=0, FilterWeights=0, stride=1):
        self.stride = stride
        Pictures, FilterWeights = self.PreProcess(Pictures, FilterWeights)
        Passes = self.CreatePasses(Pictures, FilterWeights)
        
        # 🌟 修改：使用形状的宽度维度 [3] 及 stride 来计算输出特征图的宽度
        ofmapWidth = (Pictures.shape[3] - FilterWeights.shape[3]) // self.stride + 1
        
        # 🌟 修改：向 EyerissF.Conv2d 传递 stride
        Psum = [self.EyerissF.Conv2d(ps, ofmapWidth, self.n, self.p, self.q, stride=self.stride) for ps in Passes]
        self.Reverse(Psum)
        OfMaps = self.Output()
        return self.PostProcess(OfMaps)
        
    def ReLU(self, array):
        return Activiation.ReLU(array)

    def Pooling(self, array):
        return Pooling.Pooling(array)
            
    def FullConnect(self, v1, v2, activation=1):
        return np.array(np.dot(v1, v2.T) / activation, dtype=int)
        
    def CreatePasses(self, Pictures, FilterWeights):
        self.__SetPicAndFlt__(Pictures, FilterWeights)
        return self.Conv2DMapping()
        
    def Conv2DMapping(self):
        self.__PEArrayMapping__()
        self.__PESetMapping__()
        return self.__SetPasses__()

    def __SetPasses__(self):
        import math
        Passes = []
        for batch in range( self.Pictures.shape[0] ):
            for channel in range( int(self.Pictures.shape[1]/self.r) ): 
                for ofmap in range( int(self.FilterWeights.shape[0]/self.t) ):
                    
                    # 🌟 修改：考虑 stride 计算输出高度
                    ofmapHeight = (self.Pictures.shape[2] - self.FilterWeights.shape[2]) // self.stride + 1
                    head = 0
                    total_passes = math.ceil(ofmapHeight / self.e)
                    
                    for e in range( total_passes ):
                        current_e = min(self.e, ofmapHeight - e * self.e)
                        # 🌟 修改：当前切片需要的输入行数必须由 stride 计算而来
                        ifmapEachPass = (current_e - 1) * self.stride + self.FilterWeights.shape[2]
                        tail = head + ifmapEachPass
                        
                        PicPass = self.Pictures[batch, channel*self.r:(channel+1)*self.r, head:tail, :]
                        WeightPass = self.FilterWeights[ofmap*self.t:(ofmap+1)*self.t, channel*self.r:(channel+1)*self.r, :, :]
                        Passes.append([PicPass, WeightPass])
                        
                        # 🌟 修改：下一轮切片的起点跳跃必须乘以 stride
                        head += self.e * self.stride
        return Passes
    
    def __SetMappingParameters__(self, m=0, n=0, e=0, p=0, q=0, r=0, t=0):
        self.m = m if m!=0 else self.m
        self.n = n if n!=0 else self.n
        self.p = p if p!=0 else self.p
        self.q = q if q!=0 else self.q
        self.e = e if e!=0 else self.e
        self.r = r if r!=0 else self.r
        self.t = t if t!=0 else self.t
    
    def __PEArrayMapping__(self):
        #TODO: also consider stride
        PESetHeight = self.FilterWeights.shape[2] #filter height
        PESetWidth = (self.Pictures.shape[2] - self.FilterWeights.shape[2]) // self.stride + 1 #ofmap height
        #Eyeriss only support filter height smaller than PE array height
        assert PESetHeight <= conf.EyerissHeight
        
        # 计算硬件物理上最多能同时放几个卷积核
        t = int(conf.EyerissHeight / PESetHeight) 
        
        # ⚠️ 核心修复：实际并行的卷积核数不能超过提供的权重通道数总和
        # 防止出现 int(FilterNum / t) = 0 导致 Pass 为空的问题
        t = min(t, self.FilterWeights.shape[0])
        
        #TODO: let's assume PESetW >=PEArrayWidth for now
        if PESetWidth > conf.EyerissWidth:
            #strip-mining the 2-D convolution
            fold = ( int((PESetWidth-1)/conf.EyerissWidth) + 1 )
            e = conf.EyerissWidth
            if t % fold == 0:
                t = int(t/fold)
                e = PESetWidth
        else: 
            e = PESetWidth
            
        self.__SetMappingParameters__(e=e, t=t)
        
    def __PESetMapping__(self):
        #TODO: add reusing filter, processing n ifmaps at a time
        slidingWindow = self.FilterWeights.shape[2]
        qMax = int(conf.IfmapSpad/(slidingWindow*self.n))
        q=qMax
        for q in range(qMax,0,-1):
            if self.FilterWeights.shape[1]%q == 0:
                break
        pMax = min(int(conf.PsumSpad/self.n), int(conf.FilterSpad/
                (q*slidingWindow)))
        # sometimes we don't need large p at PE level 
        # since PE array already reused it
        pMax = min(pMax, int(self.FilterWeights.shape[0]/self.t))  
        p=pMax
        for p in range(pMax,0,-1):
            if self.FilterWeights.shape[0]%p == 0:
                break
        m = self.FilterWeights.shape[0]/(self.r*q)
        self.__SetMappingParameters__(q=q,p=p)
        
        self.__FilterReuse__()
        self.__FmapReuse__()
        self.__ChannelAccumulation__()
        
        
    def __SetPicAndFlt__(self, Pictures=None, FilterWeights=None):
        if isinstance(Pictures, (np.ndarray)): self.Pictures = Pictures
        if isinstance(FilterWeights, (np.ndarray)): self.FilterWeights = FilterWeights  
        
    def __FilterReuse__(self):
        if self.n > 1:
            assert self.Pictures.shape[0]%self.n == 0
            Pictures = np.split(self.Pictures, self.n)
            Pictures = np.concatenate(Pictures, axis = 3)
                
            self.__SetPicAndFlt__(Pictures=Pictures)

    def __FmapReuse__(self):
        if self.p > 1: 
            assert self.FilterWeights.shape[0]%self.p == 0
            s = np.array(self.FilterWeights.shape)
            s[0] /= self.p
            s[3] *= self.p
            FilterWeights = np.empty(s,dtype=self.FilterWeights.dtype)
            for p in range(self.p):
                FilterWeights[:,:,:, p::self.p] = self.FilterWeights[p::self.p]
            
            self.__SetPicAndFlt__(FilterWeights = FilterWeights)

    def __ChannelAccumulation__(self):
        if self.q > 1:
            assert self.FilterWeights.shape[1]%self.q == 0
            
            s = np.array(self.Pictures.shape)
            s[1] /= self.q
            s[3] *= self.q
            Pictures = np.empty(s,dtype=self.Pictures.dtype)
            for q in range(self.q):
                Pictures[:,:,:, q::self.q] = self.Pictures[:,q::self.q,:,:]
                
            s = np.array(self.FilterWeights.shape)
            s[1] /= self.q
            s[3] *= self.q
            FilterWeights = np.empty(s,dtype=self.FilterWeights.dtype)
            for q in range(self.q):
                FilterWeights[:,:,:, q::self.q] = self.FilterWeights[:,q::self.q,:,:]
        
            self.__SetPicAndFlt__(Pictures, FilterWeights)
        

    def Reverse(self, Psum):
        import math
        # 🌟 修改：分别计算输出特征图的高度和宽度
        ofmapHeight = (self.Pictures.shape[2] - self.FilterWeights.shape[2]) // self.stride + 1
        ofmapWidth = (self.Pictures.shape[3] - self.FilterWeights.shape[3]) // self.stride + 1
        
        # 预分配完整的输出矩阵，注意传入高度和宽度
        OfMaps = np.zeros((self.Pictures.shape[0], self.FilterWeights.shape[0], 
                           ofmapHeight, ofmapWidth*self.n*self.p))
        
        total_passes = math.ceil(ofmapHeight / self.e) # 高度方向的总分片数
        num_channels = int(self.Pictures.shape[1] / self.r)
        num_ofmaps = int(self.FilterWeights.shape[0] / self.t)
        
        index = 0
        for batch in range(self.Pictures.shape[0]):
            batch_accumulator = np.zeros((self.FilterWeights.shape[0], 
                                          ofmapHeight, 
                                          ofmapWidth*self.n*self.p))
            
            for channel in range(num_channels):
                for ofmap in range(num_ofmaps):
                    width_chunks = []
                    for e in range(total_passes):
                        width_chunks.append(np.array(Psum[index]))
                        index += 1  
                        
                    stitched_row = np.concatenate(width_chunks, axis=1)
                    
                    start_t = ofmap * self.t
                    end_t = start_t + self.t
                    batch_accumulator[start_t:end_t, :, :] += stitched_row
                    
            OfMaps[batch] = batch_accumulator
            
        self.__SetOfMaps__(OfMaps)
        self.__ReverseFmapReuse__()
        self.__ReverseFilterReuse__()

    def __ReverseFmapReuse__(self):
        s = np.array(self.OfMaps.shape)
        s[1] *= self.p
        s[3] /= self.p
        OfMaps = np.zeros(s, dtype=self.OfMaps.dtype)
        for p in range(self.p):
            OfMaps[:,p::self.p] = self.OfMaps[:,:,:,p::self.p]
        self.__SetOfMaps__(OfMaps)

    def __ReverseFilterReuse__(self):
        OfMaps = np.split(self.OfMaps, self.n, axis=3)
        OfMaps = np.concatenate(OfMaps, axis = 0)
        self.__SetOfMaps__(OfMaps)

    def __SetOfMaps__(self, OfMaps):
        self.OfMaps = OfMaps

    def Output(self):
        #TODO: trace memory write
        #return self.Compress(self.ReturnImgs)
        return self.OfMaps
