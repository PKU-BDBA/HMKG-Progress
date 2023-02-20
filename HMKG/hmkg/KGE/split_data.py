import csv

def split_data(triple_path,info_path="data/info.txt"):
    Meta_All = []
    Meta_train = []
    Meta_test = []
    Meta_eva = []

    Meta_class = {}
    train_class = {}
    test_class = {}
    eva_class = {}

    ##读取文件中总数据集
    with open(triple_path, newline='', encoding='utf-8') as in_file:
        Meta = csv.reader(in_file, delimiter='\t')
        for line in Meta:
            Meta_All.append(line)

    ##使用字典格式记录每种rel的数量
    Meta_num_All = len(Meta_All)
    Meta_class[Meta_All[0][1]] = 0
    for meta_sch in Meta_All:
        if meta_sch[1] in Meta_class:
            Meta_class[meta_sch[1]] = Meta_class[meta_sch[1]] + 1
        else:
            Meta_class[meta_sch[1]] = 0

    ##获取训练、测试、评估数据集每种rel的数量
    train_ratio = 0.8
    test_ratio = 0.1
    eva_ratio = 0.1

    print(Meta_class)

    loca = 0 #用于记录每类开始的位置
    for i in Meta_class.keys():
        train_class[i] = int(train_ratio * Meta_class[i])
        test_class[i] = int(test_ratio * Meta_class[i])
        eva_class[i] = int(eva_ratio * Meta_class[i])
        start = loca #开始寻找

        for j in range(train_class[i]):
            Meta_train.append(Meta_All[start+j])
        start = start + train_class[i]

        for j in range(test_class[i]):
            Meta_test.append(Meta_All[start+j])
        start = start + test_class[i]

        for j in range(eva_class[i]):
            Meta_eva.append(Meta_All[start+j])

        loca = loca + Meta_class[i]

    print(len(Meta_train))
    print(len(Meta_test))
    print(len(Meta_eva))


    ## 按照TSV格式写入
    with open("data/" + 'TrainingSet.txt', 'wt', newline='', encoding='utf-8') as train_file:
        tsv_writer = csv.writer(train_file, delimiter='\t')
        tsv_writer.writerows(Meta_train)
    train_file.close()

    with open("data/" + 'TestSet.txt', 'wt', newline='', encoding='utf-8') as test_file:
        info_writer = csv.writer(test_file, delimiter='\t')
        info_writer.writerows(Meta_test)
    test_file.close()

    with open("data/" + 'EvaluationSet.txt', 'wt', newline='', encoding='utf-8') as eva_file:
        tsv_writer = csv.writer(eva_file, delimiter='\t')
        tsv_writer.writerows(Meta_eva)
    eva_file.close()

    #需要显示的数据集信息
    def Meta_info(All, Train, Test, Eval, pth):
        meta_tmp = []
        meta_line = []
        rel = []
        meta_tmp.append("总数据类别信息：")
        for rel in All.keys():
            meta_line = rel + " : " + str(All[rel])
            meta_tmp.append(meta_line)
        meta_tmp.append(" ")
        meta_tmp.append("训练集类别信息：")
        for rel in All.keys():
            meta_line = rel + " : " + str(Train[rel])
            meta_tmp.append(meta_line)
        meta_tmp.append(" ")
        meta_tmp.append("测试集类别信息:")
        for rel in All.keys():
            meta_line = rel + " : " + str(Test[rel])
            meta_tmp.append(meta_line)
        meta_tmp.append(" ")
        meta_tmp.append("评估集类别信息:")
        for rel in All.keys():
            meta_line = rel + " : " + str(Eval[rel])
        meta_tmp.append(meta_line)

        with open(pth, 'wt', encoding='utf-8')as f:
            for line in meta_tmp:
                f.write(line + '\n')
        f.close()

        return True

    Meta_info(Meta_class, train_class, test_class, eva_class, info_path)