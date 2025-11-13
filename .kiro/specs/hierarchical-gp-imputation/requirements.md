# Requirements Document

## Introduction

本项目旨在评估分层高斯过程回归在土木工程实验数据中的表现。系统需要处理多来源数据中的缺失值和数据差异问题,通过构建分层高斯过程回归框架实现多重插补和预测功能。

## Glossary

- **System**: 分层高斯过程回归插补系统
- **ChainedImputer**: 链式分层多重插补抽样类
- **BaseModel**: 基础回归模型(线性回归或高斯过程回归)
- **RandomEffects**: 随机效应参数,用于标识分层结构的输入列
- **MultipleImputation**: 多重插补,对缺失数据进行多次抽样填充
- **HierarchicalStructure**: 分层结构,基于组别变量建立的层次模型
- **ImputationMethod**: 插补方法,包括普通线性、分层线性、普通高斯过程、分层高斯过程

## Requirements

### Requirement 1

**User Story:** 作为研究人员,我希望使用链式多重插补类处理缺失数据,以便灵活选择不同的基础回归模型

#### Acceptance Criteria

1. THE ChainedImputer SHALL accept any regression model as the base model for multiple imputation
2. WHERE the base model has hierarchical structure, THE ChainedImputer SHALL accept a parameter name list to specify random effects
3. THE ChainedImputer SHALL determine which input columns belong to the hierarchical structure based on the random effects list
4. THE ChainedImputer SHALL handle both ordinary models and hierarchical models
5. THE ChainedImputer SHALL perform chained sampling to generate multiple imputed datasets

### Requirement 2

**User Story:** 作为研究人员,我希望使用线性回归模型进行插补和预测,以便在普通和分层两种模式下工作

#### Acceptance Criteria

1. THE System SHALL provide a linear regression model that can operate in ordinary mode
2. THE System SHALL provide a linear regression model that can operate in hierarchical mode
3. WHERE hierarchical mode is selected, THE linear regression model SHALL accept a parameter name list to establish hierarchical structure
4. THE linear regression model SHALL build hierarchical structure internally based on the specified columns
5. THE linear regression model SHALL support sampling from posterior distributions for imputation

### Requirement 3

**User Story:** 作为研究人员,我希望使用高斯过程回归模型进行插补和预测,以便在普通和分层两种模式下工作

#### Acceptance Criteria

1. THE System SHALL provide a Gaussian process regression model that can operate in ordinary mode
2. THE System SHALL provide a Gaussian process regression model that can operate in hierarchical mode
3. WHERE hierarchical mode is selected, THE Gaussian process model SHALL accept a parameter name list to establish hierarchical structure
4. THE Gaussian process model SHALL build hierarchical structure internally based on the specified columns
5. THE Gaussian process model SHALL support sampling from posterior distributions for imputation

### Requirement 4

**User Story:** 作为研究人员,我希望比较不同插补方法的直接效果,以便评估插补值的准确性

#### Acceptance Criteria

1. THE System SHALL artificially mask one input column in a dataset for testing
2. THE System SHALL perform multiple imputation using four methods: ordinary linear, hierarchical linear, ordinary GP, hierarchical GP
3. THE System SHALL compare imputed values with true values after imputation
4. THE System SHALL calculate mean squared error for each imputation method
5. THE System SHALL output comparison results in a Jupyter notebook file

### Requirement 5

**User Story:** 作为研究人员,我希望评估插补结果对下游模型的影响,以便了解插补质量对预测性能的作用

#### Acceptance Criteria

1. THE System SHALL perform multiple imputation on a dataset using four imputation methods
2. WHEN imputation is complete, THE System SHALL train support vector regression models on each imputed dataset
3. THE System SHALL use identical hyperparameter search procedures for SVR across all four methods
4. THE System SHALL output mean squared error for each method
5. THE System SHALL output R-squared score for each method
6. THE System SHALL present results in a Jupyter notebook file

### Requirement 6

**User Story:** 作为研究人员,我希望测试分层模型在未知组别下的预测能力,以便评估模型的泛化性能

#### Acceptance Criteria

1. THE System SHALL use a dataset without missing values for prediction testing
2. THE System SHALL perform prediction using four methods: ordinary linear, hierarchical linear, ordinary GP, hierarchical GP
3. WHEN making predictions, THE System SHALL treat test sample groups as unknown
4. THE System SHALL compare prediction errors across the four methods
5. THE System SHALL compare prediction uncertainty across the four methods
6. THE System SHALL observe performance changes of hierarchical models when facing unknown groups
7. THE System SHALL present results in a Jupyter notebook file
