import numpy as np


def loaddata():
    x=np.arange(0,10,1)
    y=np.array([1]*3 + [-1]*3 + [1]*3 + [-1])
    return x,y

def initWeights():
    w = np.array([0.1]*10) 
    return w

def determineCurrentLoopThreshold(x,y,w):
    errorRate1 = 0.0 # 2种基本形式的分类器，大于和小于
    errorRate2 = 0.0
    errorRateThresholdDictionary1 = dict()
    errorRateThresholdDictionary2 = dict()
    currentThreshold = 0
    G1 = []
    G2 = []

    for currentThreshold in np.arange(min(x)+1.5,max(x),1):
        # print(currentThreshold)
        errorRate1 =0
        errorRate2=0
        G1 = [1 if i<currentThreshold else -1 for i in x]
        G2 = [-1 if i<currentThreshold else 1 for i in x]
        errorRate1 = ( (G1!=y)*w).sum()
        errorRate2 = ( (G2!=y)*w).sum()
        errorRateThresholdDictionary1[errorRate1] = currentThreshold
        errorRateThresholdDictionary2[errorRate2] = currentThreshold

    errorRate1 = min(  list(errorRateThresholdDictionary1.keys())  )
    threshold1 = list(errorRateThresholdDictionary1.values()) [ list(errorRateThresholdDictionary1.values()).index(errorRateThresholdDictionary1[errorRate1])    ]
    
    errorRate2 = min(  list(errorRateThresholdDictionary2.keys())  )
    threshold2 = list(errorRateThresholdDictionary2.values()) [ list(errorRateThresholdDictionary2.values()).index(errorRateThresholdDictionary2[errorRate2])    ]



    if errorRate1 < errorRate2:
        G1 = [1 if i<threshold1 else -1 for i in x]
        return errorRate1,threshold1, G1, 1
    else:
        G2 = [-1 if i<threshold2 else 1 for i in x]
        return errorRate2,threshold2, G2, 2
 


if __name__ == '__main__':

    alpha = []
    thresholdList = []
    basicClassifierIDList = []

    # 加载数据
    x,y = loaddata()
    # print(x,y)

    # 初始化权重
    w = initWeights()
    # print(w)
    loopCnt=0
    GFinal = np.array([0]*10,dtype=int)
    fx = np.array([0]*10,dtype=float)

    while(1):
        # 决定当前数据权重分布上，当前分类器分类误差率最小阈值和误差率
        errorRate,threshold,G,basicClassifierID = determineCurrentLoopThreshold(x,y,w)
        thresholdList.append(threshold)
        G = np.array(G)
        basicClassifierIDList.append(basicClassifierID)

        # 判断是否有误分类点
        loopCnt+=1
        print(loopCnt, errorRate, threshold,basicClassifierID)
        print(y)
        print(GFinal,'-------------------')
        if((y==GFinal).all()):
            break

        # 计算当前权重分布，分类器的系数
        alpha.append( 0.5*np.log( (1-errorRate)/errorRate ) ) 
        alphaCurrent = alpha[len(alpha)-1]

        # 更新强分类器
        fx += alphaCurrent*G
        GFinal = np.sign(fx)

        # 更新权重系数
        G = np.array(G)        
        w = w*np.exp(-alphaCurrent*y*G)/(w*np.exp(-alphaCurrent*y*G)).sum() 
        a=0   
    # 最后返回alpha列表和弱分类器阈值列表
    alpha,fx,GFinal
