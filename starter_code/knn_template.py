from numpy import *
import matplotlib.pyplot as plt
import operator

def createDataSet():
    """创建一个简单的测试数据集"""
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels

def file2matrix(filename):
    """从文件中读取数据并转换为矩阵和标签"""
    fr = open(filename)
    array_olines = fr.readlines()
    number_lines = len(array_olines)
    return_mat = zeros((number_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_olines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector

def autoNorm(dataSet):
    """数据归一化处理"""
    minVals = dataSet.min(0)  # 计算每列的最小值
    maxVals = dataSet.max(0)  # 计算每列的最大值
    ranges = maxVals - minVals  # 计算每列数据的范围
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 获取数据集的行数
    normDataSet = dataSet - tile(minVals, (m, 1))  # 减去最小值
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 除以范围，归一化到[0,1]
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    """KNN分类核心算法"""
    # 计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    
    # 排序距离
    sortedDistIndicies = distances.argsort()
    
    # 选择k个最近邻
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    # 排序并返回最可能的类别
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    """测试KNN分类算法的准确率"""
    hoRatio = 0.50  # 测试集比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读取数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化
    m = normMat.shape[0]  # 数据总数
    numTestVecs = int(m * hoRatio)  # 测试集数量
    errorCount = 0.0  # 错误计数
    
    # 测试分类效果
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], 
                                    datingLabels[numTestVecs:m], 3)
        print(f"预测结果: {classifierResult}, 实际结果: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    
    # 输出错误率
    print(f"错误率: {errorCount / float(numTestVecs)}")
    return errorCount / float(numTestVecs)

def classify_person():
    """交互式输入特征，进行分类预测"""
    resultList = ['一点也不喜欢', '有点喜欢', '非常喜欢']
    
    # 获取用户输入
    percentTats = float(input("玩视频游戏所耗时间百分比?"))
    ffMiles = float(input("每年获得的飞行常客里程数?"))
    iceCream = float(input("每周消费的冰淇淋公升数?"))
    
    # 读取数据并训练模型
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 准备输入数据
    inArr = array([ffMiles, percentTats, iceCream])
    
    # 归一化输入数据
    norminArr = (inArr - minVals) / ranges
    
    # 进行分类预测
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    
    # 输出结果
    print(f"你可能{resultList[classifierResult - 1]}这个人")

# 死循环调用交互式分类函数
if __name__ == "__main__":
    while True:
        classify_person()
        # 询问是否继续
        again = input("是否继续? (y/n): ")
        if again.lower() != 'y':
            break