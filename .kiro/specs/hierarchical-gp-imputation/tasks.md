# Implementation Plan

- [x] 1. 创建项目结构和基础模型接口



  - 创建项目目录结构: models/, imputation/, experiments/
  - 定义BaseModel抽象基类,包含fit, predict, sample方法
  - 创建requirements.txt列出所有依赖库
  - _Requirements: 1.1, 2.1, 3.1_

- [x] 2. 实现普通线性回归模型

  - 实现LinearRegression类,继承BaseModel
  - 使用BayesianRidge实现fit方法
  - 实现predict方法,支持return_std参数
  - 实现sample方法,从后验分布抽样
  - _Requirements: 2.1, 2.5_


- [ ] 3. 实现分层线性回归模型
  - 在LinearRegression类中添加分层模式支持
  - 接收random_effects参数列表
  - 使用statsmodels.MixedLM实现分层结构
  - 实现分层模式下的sample方法
  - _Requirements: 2.2, 2.3, 2.4, 2.5_



- [ ] 4. 实现普通高斯过程回归模型
  - 实现GaussianProcessRegression类,继承BaseModel
  - 使用GaussianProcessRegressor实现fit方法
  - 实现predict方法,支持return_std参数
  - 实现sample方法,从后验分布抽样

  - _Requirements: 3.1, 3.5_

- [ ] 5. 实现分层高斯过程回归模型
  - 在GaussianProcessRegression类中添加分层模式支持
  - 接收random_effects参数列表


  - 实现自定义核函数或使用GPy支持组别特定随机效应
  - 实现分层模式下的sample方法,处理未知组别情况
  - _Requirements: 3.2, 3.3, 3.4, 3.5_

- [ ] 6. 实现ChainedImputer类
  - 创建ChainedImputer类,接收base_model和random_effects参数
  - 实现初始化逻辑,用均值填充缺失值


  - 实现链式插补迭代逻辑
  - 对每个有缺失的变量依次训练模型并抽样
  - 实现fit_transform方法,返回多个插补数据集
  - 添加收敛性检查逻辑
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 7. 实现实验一: 直接插补效果比较


  - 创建experiment_1_direct_imputation.ipynb
  - 加载或生成测试数据集
  - 随机遮盖一个输入列的部分数据
  - 使用四种方法(普通线性、分层线性、普通GP、分层GP)进行插补
  - 计算每种方法的MSE
  - 创建对比可视化图表
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_




- [ ] 8. 实现实验二: 下游模型影响评估
  - 创建experiment_2_downstream_impact.ipynb
  - 使用与实验一相同的数据集和遮盖策略
  - 使用四种方法进行插补
  - 对每个插补数据集训练SVR模型
  - 使用GridSearchCV进行超参数搜索
  - 计算并输出每种方法的MSE和R²
  - 创建性能对比可视化
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 9. 实现实验三: 未知组别预测测试
  - 创建experiment_3_unknown_group.ipynb
  - 加载无缺失值的完整数据集
  - 划分训练集和测试集,测试集包含未知组别
  - 使用四种方法进行训练和预测
  - 计算预测误差和预测不确定性
  - 对比分层模型在未知组别下的性能变化
  - 创建结果对比可视化
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_
