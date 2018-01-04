# encoding: utf-8 
from numpy import *  
import operator  
  
#样本数据集创建函数  
def creatDataSet():  
    dataSet = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])  
    labels = ['A','A','B','B']  
    return dataSet,labels  
  
  
#k-NN简单实现函数 
# inX表示待分类的输入特征向量， 
# dataSet为样本集的特征， 
# labels为样本集对应每一个样本的分类标签， 
# k为选择最近距离的样本的数目(kNN) 
def classify0(inX,dataSet,labels,k):  
  
    #求出样本集的行数，也就是labels标签的数目  
    dataSetSize = dataSet.shape[0]  
  
    #构造输入值和样本集的差值矩阵  
    diffMat = tile(inX,(dataSetSize,1)) - dataSet  
  
    #计算欧式距离  
    sqDiffMat = diffMat**2  
    sqDistances = sqDiffMat.sum(axis=1)  
    distances = sqDistances**0.5  
  
    #求距离从小到大排序的序号  
    sortedDistIndicies = distances.argsort()  
  
    #对距离最小的k个点统计对应的样本标签  
    classCount = {}  
    for i in range(k):  
        #取第i+1近邻的样本对应的类别标签  
        voteIlabel = labels[sortedDistIndicies[i]]  
        #以标签为key，标签出现的次数为value将统计到的标签及出现次数写进字典  
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  
  
    #对字典按value从大到小排序  
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  
  
    #返回排序后字典中最大value对应的key  
    return sortedClassCount[0][0]  
  
#从文本文件中解析数据  
def file2matrix(filename):  
    fr = open(filename)  
      
    #按行读取整个文件的内容  
    arrayOfLines = fr.readlines()  
    print(arrayOfLines)
    exit()  
    #求文件行数，即样本总数  
    numberOfLines = len(arrayOfLines)  
  
    #初始化返回数组及列表  
    returnMat = zeros((numberOfLines,3))  
    classLabelVector = []  
      
    index = 0  
    for line in arrayOfLines:  
        #截取掉每一行的回车符  
        line = line.strip()  
        #将一行数据根据制表符分割成包含多个元素（特征）的列表  
        listFromLine = line.split('\t')  
        #分割后的特征数据列表存入待返回的numpy数组  
        returnMat[index,:] = listFromLine[0:3]  
        #将文件每一行最后一列的数据存入待返回的分类标签列表  
        classLabelVector.append(int(listFromLine[-1]))  
        #控制待返回的numpy数组进入下一行  
        index+=1  
    return returnMat,classLabelVector  
  
#特征变量归一化  
def autoNorm(dataSet):  
  
    #取出每一列的最小值，即每一个特征的最小值  
    minVals = dataSet.min(0)  
  
    #取出每一列的最大值，即每一个特征的最大值  
    maxVals = dataSet.max(0)  
  
    #每一个特征变量的变化范围  
    ranges = maxVals - minVals  
  
    #初始化待返回的归一化特征数据集  
    normDataSet = zeros(shape(dataSet))  
  
    #特征数据集行数，即样本个数  
    m = dataSet.shape[0]  
  
    #利用tile()函数构造与原特征数据集同大小的矩阵，并进行归一化计算  
    normDataSet = dataSet - tile(minVals,(m,1))  
    normDataSet = normDataSet/tile(ranges,(m,1))  
  
    return normDataSet,ranges,minVals  
  
#分类器测试  
def datingClassTest():  
  
    #给定用于测试分类器的样本比例  
    hoRatio = 0.10  
  
    #调用文本数据解析函数  
    datingDataMat,datingLabels = file2matrix('E:/files/share/dating_test_set.txt')  
  
    #调用特征变量归一化函数  
    normMat,ranges,minVals = autoNorm(datingDataMat)  
  
    #归一化的特征数据集行数，即样本个数  
    m = normMat.shape[0]  
  
    #用于测试分类器的测试样本个数  
    numTestVecs = int(m*hoRatio)  
      
    #初始化分类器犯错样本个数  
    errorCount = 0.0  
      
    for i in range(numTestVecs):  
        #以测试样本作为输入，测试样本之后的样本作为训练样本，对测试样本进行分类  
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)  
  
        #对比分类器对测试样本预测的类别和其真实类别  
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult,datingLabels[i]))  
  
        #统计分类出错的测试样本数  
        if (classifierResult != datingLabels[i]):  
            errorCount+=1.0  
  
    #输出分类器错误率          
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))  
  
#约会网站对新输入分类  
def classifyPerson():  
  
    #定义一个存储了三个字符串的列表，分别对应不喜欢，一般喜欢，很喜欢  
    resultList = ['not at all','in small dose','in large dose']  
  
    #用户输入三个特征变量，并将输入的字符串类型转化为浮点型  
    ffMiles = float(input("frequent flier miles earned per year:"))  
    percentats = float(input("percentage of time spent playing video games:"))  
    iceCream = float(input("liters of ice cream consumed per year:"))  
  
    #调用文本数据解析函数  
    datingDataMat,datingLabels = file2matrix('E:/files/share/dating_test_set.txt')  
  
    #调用特征变量归一化函数  
    normMat,ranges,minVals = autoNorm(datingDataMat)  
  
    #将输入的特征变量构造成特征数组（矩阵）形式  
    inArr = array([ffMiles,percentats,iceCream])  
  
    #调用kNN简单实现函数  
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)  
  
    #将分类结果由数字转换为字符串  
    print("You will probably like this person",resultList[classifierResult - 1])  


classifyPerson()
