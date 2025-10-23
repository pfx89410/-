from numpy import*
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group=array([[1.0,1,1],[1.0,1.0],[0,0],[0,0.1]])
    labels=["A","A","B","B"]
    return group,labels

def file2matrix(filename):
    fr=open(filename)   #打开文件
    array_olines=fr.readlines() #从文件中读取一行
    number_lines=len(array_olines) 
    return_mat=zeros((number_lines,3))
    class_label_vector=[]
    index=0
    for line in array_olines:
        line=line.strip()
        list_from_line=line.split('\t')
        return_mat[index,:]=list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index+=1
    return return_mat,class_label_vector


file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
dating_data_mat, dating_labels = file2matrix(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

fig=plt.figure()
ax=fig.add_subplot(111)  
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2])
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2],15.0*array(dating_labels),15.0*array(dating_labels))
ax.set_xlabel('第二特征')
ax.set_ylabel('第三特征')
ax.set_title('特征关系散点图')
#plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 补充：KNN核心分类函数
def classify0(inX, dataSet, labels, k):
    # 计算待分类样本与训练集所有样本的欧氏距离
    dataSetSize = dataSet.shape[0]  # 训练集样本数量
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 差值矩阵（扩展inX与训练集做差）
    sqDiffMat = diffMat **2  # 差值平方
    sqDistances = sqDiffMat.sum(axis=1)  # 按行求和（距离平方）
    distances = sqDistances** 0.5  # 开方得欧氏距离
    
    # 按距离从小到大排序，返回索引
    sortedDistIndicies = distances.argsort()
    
    # 统计前k个近邻的标签
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 获取第i个近邻的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 标签计数
    
    # 按标签出现次数排序，返回最多的标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 补充：测试算法准确率
def datingClassTest():
    hoRatio = 0.10  # 10%数据作为测试集
    datingDataMat, datingLabels = file2matrix(file_path)  # 加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化
    m = normMat.shape[0]  # 总样本数
    numTestVecs = int(m * hoRatio)  # 测试集样本数
    errorCount = 0.0  # 错误计数
    
    for i in range(numTestVecs):
        # 用后90%作为训练集，前10%作为测试集
        classifierResult = classify0(
            normMat[i, :],  # 测试样本
            normMat[numTestVecs:m, :],  # 训练集特征
            datingLabels[numTestVecs:m],  # 训练集标签
            3  # k=3
        )
        print(f"预测结果: {classifierResult}, 实际结果: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    
    print(f"总错误率: {errorCount / float(numTestVecs)}")


# 补充：交互式分类函数
def classify_person():
    resultList = ['不喜欢', '一般', '很喜欢']  # 标签1/2/3对应的中文
    
    # 输入用户特征
    ffMiles = float(input("每年飞行里程数？"))
    percentTats = float(input("玩游戏时间占比(%)？"))
    iceCream = float(input("每周冰淇淋消耗量(公升)？"))
    
    # 加载并预处理数据
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 构造待预测样本并归一化
    inArr = array([ffMiles, percentTats, iceCream])
    normInArr = (inArr - minVals) / ranges  # 使用训练集的归一化参数
    
    # 预测并输出结果
    classifierResult = classify0(normInArr, normMat, datingLabels, 3)
    print(f"对这个人的印象：{resultList[classifierResult - 1]}")


# 程序入口：死循环调用交互式分类
if __name__ == "__main__":
    while True:
        classify_person()
        print("------------------")  # 分隔多次查询