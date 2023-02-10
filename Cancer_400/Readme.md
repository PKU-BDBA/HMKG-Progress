功能规划列表

### HMKG库

- 完整知识图谱构建

- 部分子图构建（支持多种输入）
- 知识图谱数据统计，导出统计表
- 映射至Neo4j图数据库
- 代谢物相关信息查找（单个、组合）
- 代谢物反向查找（单个、组合）
- 结果可视化
- KGE模型训练
- Link Prediction

### Meta2Vec库

- 输入HMDB_id，输出向量
- 输入两个HMDB_id，计算相似度
- 输入一组HMDB_id，进行k-means聚类，umap分析



### 当前函数设计

查看[Cancer400_KGE.py](https://github.com/PKU-BDBA/HMKG-Progress/blob/main/Cancer_400/Cancer400_KGE.py)

python main.py运行

