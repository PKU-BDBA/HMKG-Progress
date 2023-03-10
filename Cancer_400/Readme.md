## 功能规划列表 （持续更新）

### HMKG库

- [ ] ~~完整知识图谱构建~~
- [ ] ~~下载HMDB数据~~
- [ ] ~~构建三元组~~
- [ ] ~~部分子图构建（支持多种输入）~~
- [x] 知识图谱数据统计，导出统计表
- [x] 三元组统计信息可视化，知识图谱可视化
- [ ] 映射至Neo4j图数据库
- [x] 代谢物相关信息查找（单个、组合）
- [x] 代谢物反向查找（单个、组合）
- [x] 查找结果可视化，查找结果导出
- [x] pykeen三元组构建
- [x] 保存id映射文件
- [x] KGE模型训练
- [x] KGE模型测试
- [x] 保存所有Embedding 
- [x] KGE模型训练pipeline
- [x] 获得所有HMDB向量
- [x] 获得所有需要的类别的向量
- [ ] Link Prediction

### Meta2Vec库
查看[meta2vec](https://github.com/YuxingLu613/meta2vec)

[pypi地址](https://pypi.org/project/meta2vec/0.0.4/)
- [x] 输入HMDB_id，输出向量
- [x] 输入两个HMDB_id，计算相似度
- [x] 输入一组HMDB_id，umap分析



### 当前函数设计

查看[HMKG.py](https://github.com/PKU-BDBA/HMKG-Progress/blob/main/Cancer_400/HMKG.py)

python main.py运行

