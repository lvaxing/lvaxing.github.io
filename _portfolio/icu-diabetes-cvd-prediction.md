<!--
 * @Author: Lvaxing axing_lv@163.com
 * @Date: 2026-01-15 14:39:52
 * @LastEditors: Lvaxing axing_lv@163.com
 * @LastEditTime: 2026-01-15 16:00:11
 * @FilePath: \undefinede:\github repository\myintro\_portfolio\icu-diabetes-cvd-prediction.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->



---
title: "ICU糖尿病人群CVD预测模型：LASSO-XGBoost"
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
---

## 项目背景
在全球人口老龄化背景下，ICU患者中糖尿病的患病率逐年上升，而心血管疾病（CVD）是ICU糖尿病患者的主要死亡原因。早期预测CVD风险对ICU患者至关重要，有助于临床决策和靶向干预。

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
![LASSO回归路径](/images/portfolio/icu-diabetes-cvd-prediction/LASSO_path.png)
使用LASSO回归，76个特征最终共57个特征得到保留。

### 混淆矩阵
![混淆矩阵](/images/portfolio/icu-diabetes-cvd-prediction/Confusion_Matrix_Model_best.png)
混淆矩阵清晰展示了模型在测试集上的分类性能，直观反映了真阳性、真阴性、假阳性和假阴性的数量，帮助评估模型的临床应用价值。

### 模型核心表现
![模型核心表现](/images/portfolio/icu-diabetes-cvd-prediction/Metrics_Comparison_Two_Models.png)
使用Accuracy、Precision、Recall、F1-score、Kappa、ROC AUC和AUPRC多维度评价模型表现。

### ROC曲线
![ROC曲线](/images/portfolio/icu-diabetes-cvd-prediction/ROC_Comparison.png)
LASSO-XGBoost预测CVD的AUROC=0.713。

![亚型ROC曲线](/images/portfolio/icu-diabetes-cvd-prediction/ROC_Comparison_cvdsubtypes.png)
LASSO-XGBoost预测HF的AUROC=0.710，MI的AUROC=0.685，IS的AUROC=0.655。

### PRC曲线
![PRC曲线](/images/portfolio/icu-diabetes-cvd-prediction/PRC_Comparison.png)
LASSO-XGBoost预测CVD的AUPRC=0.539。

### 特征重要性
![VIMP](/images/portfolio/icu-diabetes-cvd-prediction/featureimportance_Test.png)
VIMP显示，他汀药物、肾病、硝酸甘油和轻度肝病是影响CVD风险的最关键因素。

![SHAP summary](/images/portfolio/icu-diabetes-cvd-prediction/shap_summary_plot.png)
SHAP显示，BUN、年龄、硝酸甘油和他汀药物是影响CVD风险的最关键因素。

![SHAP剂量效应关系：BUN](/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_bun.png)
![SHAP剂量效应关系：年龄](/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_age.png)
![SHAP剂量效应关系：硝酸甘油](/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_nitrates.png)
![SHAP剂量效应关系：他汀](/images/portfolio/icu-diabetes-cvd-prediction/dependence_plot_statin.png)


## 结论与展望
1. 本项目使用LASSO-XGBoost模型成功预测ICU糖尿病患者的CVD风险，ROC-AUC达到0.713。
2. 使用他汀药物和硝酸甘油药物是ICU糖尿病患者CVD风险的核心因素，需重点进行临床干预。
3. 未来工作可纳入更多临床数据（如基因数据、影像数据），并探索深度学习模型以进一步提高预测准确性。






