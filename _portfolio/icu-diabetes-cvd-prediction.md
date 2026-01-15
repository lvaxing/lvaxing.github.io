---
title: "ICU糖尿病人群CVD预测模型: LASSO-XGBoost"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/icu-diabetes-cvd-prediction
date: 2026-01-15
excerpt: "基于LASSO特征选择和XGBoost模型构建ICU糖尿病患者心血管疾病（CVD）风险预测模型，ROC-AUC达0.713。"
header:
  teaser: /images/portfolio/icu-diabetes-cvd-prediction/ROC_Comparison_cvdsubtypes.png
tags:
  - 机器学习
  - 医疗数据分析
  - 风险预测
  - XGBoost
  - LASSO
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: XGBoost
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
  - name: shap
---


## 项目背景
心血管疾病（CVD）是ICU糖尿病患者的主要死亡原因，早期预测CVD风险对ICU患者至关重要，有助于临床决策和靶向干预。

## 核心实现

### 数据清洗
```python
# 数据清洗
df = df.dropna()
df = df.drop(columns=['patient_id'])
```

### 特征工程与LASSO特征选择
```python
# 处理分类特征
df = pd.get_dummies(df, prefix=True)

# LASSO特征选择
X = df.drop(columns=['cvd'])
Y = df['cvd']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, Y)

selected_features = X.columns[lasso.coef_ != 0]
X_selected = X[selected_features]
```

### XGBoost模型构建与评估
```python
# 数据集划分
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42)

# 训练XGBoost模型
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, Y_train)

# 模型评估
Y_proba = xgb_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(Y_test, Y_proba)
print(f'ROC-AUC: {roc_auc:.2f}')
```

## 分析结果
### 特征选择
<img src="/images/portfolio/icu-diabetes-cvd-prediction/LASSO_path.png"
     alt="LASSO回归路径"
     style="width:70%; max-width:800px; display:block; margin:auto;">
采用 LASSO（Least Absolute Shrinkage and Selection Operator）回归进行特征选择，通过对回归系数施记 𝐿1 正则化，实现变量压缩与冗余特征剔除。在交叉验证确定的最优惩罚参数𝜆下，原始纳入的 76 个候选特征中最终保留 57 个非零系数特征。这一过程有效降低了特征维度，缓解多重共线性问题，为后续 XGBoost 模型训练提供了更具判别力和稳定性的输入特征集合。


### 混淆矩阵
<img src="/images/portfolio/icu-diabetes-cvd-prediction/Confusion_Matrix_Model_best.png"
     alt="混淆矩阵"
     style="width:70%; max-width:800px; display:block; margin:auto;">
混淆矩阵系统性展示了模型在独立测试集上的分类结果，包括真阳性（TP）、真阴性（TN）、假阳性（FP）及假阴性（FN）的分布情况。结果显示，模型在维持较高真阴性识别能力的同时，能够较为有效地识别 CVD 高风险患者。该结果为模型在 ICU 场景下进行风险分层与辅助临床决策提供了直观依据，尤其有助于评估误判所可能带来的临床后果。


### 模型核心表现
<img src="/images/portfolio/icu-diabetes-cvd-prediction/Metrics_Comparison_Two_Models.png"
     alt="模型核心表现"
     style="width:70%; max-width:800px; display:block; margin:auto;">
从多个互补指标对模型性能进行综合评估，包括 Accuracy、Precision、Recall、F1-score、Cohen’s Kappa、ROC AUC 及 AUPRC。


### ROC曲线
<img src="/images/portfolio/icu-diabetes-cvd-prediction/ROC_Comparison.png"
     alt="ROC曲线"
     style="width:70%; max-width:800px; display:block; margin:auto;">
在总体 CVD 预测任务中，LASSO-XGBoost 模型的 AUROC 达到 0.713，表明模型具备较好的区分 CVD 与非 CVD 患者的能力。ROC 曲线在不同阈值下体现了灵敏度与特异度之间的权衡，为临床应用中阈值选择提供了依据。


<img src="/images/portfolio/icu-diabetes-cvd-prediction/ROC_Comparison_cvdsubtypes.png"
     alt="亚型ROC曲线"
     style="width:70%; max-width:800px; display:block; margin:auto;">
在 CVD 亚型分析中，模型对不同结局的预测性能存在一定差异：心力衰竭（HF）的 AUROC 为 0.710，心肌梗死（MI）为 0.685，缺血性卒中（IS）为 0.655。结果提示模型对心源性事件的识别能力相对更强，而对卒中结局的预测仍存在进一步优化空间。


### PRC曲线
<img src="/images/portfolio/icu-diabetes-cvd-prediction/PRC_Comparison.png"
     alt="PRC曲线"
     style="width:70%; max-width:800px; display:block; margin:auto;">
在类别不平衡的背景下，Precision-Recall 曲线更能反映模型对阳性事件的识别能力。LASSO-XGBoost 在 CVD 预测中的 AUPRC 为 0.539，提示模型在维持较高召回率的同时，仍具备合理的精确度水平，具有一定的临床实用潜力，尤其适用于高风险人群的早期筛查。


### 特征重要性
<img src="/images/portfolio/icu-diabetes-cvd-prediction/featureimportance_Test.png"
     alt="VIMP"
     style="width:70%; max-width:800px; display:block; margin:auto;">
基于 XGBoost 的变量重要性（VIMP）分析显示，他汀类药物使用、肾病史、硝酸甘油应用以及轻度肝病在模型决策中占据核心地位。这些特征在模型分裂节点中的高频使用，反映了其在区分 CVD 风险方面的显著贡献，亦与既往临床认知高度一致。


<img src="/images/portfolio/icu-diabetes-cvd-prediction/shap_summary_plot.png"
     alt="SHAP summary"
     style="width:70%; max-width:800px; display:block; margin:auto;">
SHAP（SHapley Additive exPlanations）分析进一步揭示了特征对模型预测的方向性和个体层面影响。结果显示，BUN、年龄、硝酸甘油使用以及他汀类药物是驱动 CVD 风险预测的关键变量。其中，较高的 BUN 水平和年龄增长通常与更高的 CVD 预测风险相关。


<img src="/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_bun.png"
     alt="SHAP剂量效应关系：BUN"
     style="width:70%; max-width:800px; display:block; margin:auto;">
<img src="/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_age.png"
     alt="SHAP剂量效应关系：年龄"
     style="width:70%; max-width:800px; display:block; margin:auto;">
<img src="/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_nitrates.png"
     alt="SHAP剂量效应关系：硝酸甘油"
     style="width:70%; max-width:800px; display:block; margin:auto;">
<img src="/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_statin.png"
     alt="SHAP剂量效应关系：他汀"
     style="width:70%; max-width:800px; display:block; margin:auto;">
SHAP 依赖图展示了关键变量的剂量–效应关系：BUN 与年龄呈现出随数值升高而 CVD 风险逐渐增加的趋势；硝酸甘油和他汀类药物的影响则体现了治疗指征与潜在基础心血管风险之间的复杂交互关系。这些结果增强了模型的可解释性，为临床医生理解模型预测逻辑及其潜在应用场景提供了重要支持。

## 结论与展望
1. 本项目使用LASSO-XGBoost模型成功预测ICU糖尿病患者的CVD风险，ROC-AUC达到0.713。
2. 使用他汀药物和硝酸甘油药物是ICU糖尿病患者CVD风险的核心因素，需重点进行临床观察与干预。
3. 未来工作可纳入更多临床数据（如基因数据、影像数据），并探索深度学习模型以进一步提高预测准确性。







