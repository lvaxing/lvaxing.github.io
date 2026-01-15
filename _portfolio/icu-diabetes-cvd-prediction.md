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
layout: single
classes: portfolio-narrow
---

<!-- 自定义CSS美化 -->
<style>
/* 整体样式 */
.portfolio-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px 30px;
  font-family: "Helvetica Neue", Arial, sans-serif;
  line-height: 1.6;
  color: #333;
}

/* 标题样式 */
.project-title {
  font-size: 2.2rem;
  color: #2c3e50;
  border-bottom: 3px solid #3498db;
  padding-bottom: 10px;
  margin-bottom: 30px;
}

/* 技术栈标签 */
.tech-tag {
  display: inline-block;
  background-color: #e3f2fd;
  color: #2196f3;
  padding: 5px 12px;
  border-radius: 20px;
  margin: 0 5px 8px 0;
  font-size: 0.9rem;
}

/* 代码块样式 */
.code-block {
  background-color: #f8f9fa;
  border-left: 4px solid #3498db;
  padding: 15px;
  border-radius: 4px;
  margin: 20px 0;
  overflow-x: auto;
}

/* 图片样式 */
.result-img {
  max-width: 80%;
  margin: 15px auto;
  display: block;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
}

.result-img:hover {
  transform: scale(1.02);
}

/* 章节标题 */
.section-title {
  font-size: 1.5rem;
  color: #2980b9;
  margin: 30px 0 15px 0;
}

/* 结论区块 */
.conclusion-box {
  background-color: #f5fafe;
  padding: 20px;
  border-radius: 8px;
  margin-top: 30px;
}
</style>

<div class="portfolio-container">
  <h1 class="project-title">ICU糖尿病人群CVD预测模型: LASSO-XGBoost</h1>

  <!-- 技术栈展示 -->
  <div>
    <h3 style="margin-bottom: 10px;">核心技术栈</h3>
    {% for tech in page.tech_stack %}
      <span class="tech-tag">{{ tech.name }}</span>
    {% endfor %}
  </div>

  <!-- 项目背景 -->
  <h2 class="section-title">项目背景</h2>
  <p>心血管疾病（CVD）是ICU糖尿病患者的主要死亡原因，早期预测CVD风险对ICU患者至关重要，有助于临床决策和靶向干预。</p>

  <!-- 核心实现 -->
  <h2 class="section-title">核心实现</h2>

  <!-- 数据清洗 -->
  <h3>数据清洗</h3>
  <div class="code-block">
<pre>
# 数据清洗
df = df.dropna()
df = df.drop(columns=['patient_id'])
</pre>
  </div>

  <!-- 特征工程与LASSO特征选择 -->
  <h3>特征工程与LASSO特征选择</h3>
  <div class="code-block">
<pre>
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# 特征拆分（需提前定义num_cols/cat_cols/other_cvd_col）
scaler = StandardScaler()
x_df = data[num_cols + cat_cols + other_cvd_col]
feature_names = x_df.columns.tolist()

# 数据集划分
x_train_df, x_test_df, y_train, y_test = train_test_split(
    x_df, y, test_size=0.2, random_state=2025, stratify=y
)

# 分类变量独热编码
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)  
x_train_cat_encoded = ohe.fit_transform(x_train_df[cat_cols])
x_test_cat_encoded = ohe.transform(x_test_df[cat_cols])
cat_encoded_cols = ohe.get_feature_names_out(cat_cols).tolist()

X_train_cat_df = pd.DataFrame(
    x_train_cat_encoded, columns=cat_encoded_cols, index=x_train_df.index
).astype(np.float32)
X_test_cat_df = pd.DataFrame(
    x_test_cat_encoded, columns=cat_encoded_cols, index=x_test_df.index
).astype(np.float32)

# 数值变量标准化
scaler = StandardScaler()
X_train_scaled_num = scaler.fit_transform(x_train_df[num_cols])
X_test_scaled_num = scaler.transform(x_test_df[num_cols])  # 训练集scaler复用

X_train_num_df = pd.DataFrame(
    X_train_scaled_num, columns=num_cols, index=x_train_df.index
).astype(np.float32)
X_test_num_df = pd.DataFrame(
    X_test_scaled_num, columns=num_cols, index=x_test_df.index
).astype(np.float32)

# 特征合并
X_train_scaled_df = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
X_test_scaled_df = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

# LASSO特征选择
X = df.drop(columns=['cvd'])
Y = df['cvd']
X_scaled = scaler.fit_transform(X)  # 全局特征标准化

lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, Y)

selected_features = X.columns[lasso.coef_ != 0]
X_selected = X[selected_features]
print(f"LASSO筛选后保留特征数: {len(selected_features)}")
</pre>
  </div>

  <!-- XGBoost模型构建与评估 -->
  <h3>XGBoost模型构建与评估</h3>
  <div class="code-block">
<pre>
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score

# 最优参数（需提前通过调参得到bestparams）
xgb_params_best = {
    "learning_rate": bestparams["eta"],
    "booster": bestparams["booster"],
    "colsample_bytree": bestparams["colsample_bytree"],
    "colsample_bynode": bestparams["colsample_bynode"],
    "gamma": bestparams["gamma"],
    "reg_lambda": bestparams["lambda"],
    "min_child_weight": bestparams["min_child_weight"],
    "max_depth": int(bestparams["max_depth"]),
    "subsample": bestparams["subsample"],
    "objective": "binary:logistic",
    "rate_drop": bestparams["rate_drop"],
    "n_estimators": int(bestparams["num_boost_round"]),
    "verbosity": 0,
    "eval_metric": "auc",
    "base_score": 0.5,
    "random_state": 2025
}

# 模型训练
model = xgb.XGBClassifier(**xgb_params_best)
model.fit(X_train_scaled_df[selected_features], y_train)

# 模型预测与评估
y_pred_proba = model.predict_proba(X_test_scaled_df[selected_features])[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print(f'LASSO-XGBoost模型ROC-AUC: {auc:.3f}')
</pre>
  </div>

  <!-- 分析结果 -->
  <h2 class="section-title">分析结果</h2>
  <h3>特征选择</h3>
  <p>采用LASSO回归进行特征选择，在最优惩罚参数下，从76个候选特征中保留57个非零系数特征，有效降低维度、缓解多重共线性。</p>
  <img src="/images/portfolio/icu-diabetes-cvd-prediction/LASSO_path.png" 
       alt="LASSO回归路径" class="result-img">

  <h3>模型核心表现</h3>
  <p>模型总体ROC-AUC达0.713，其中心力衰竭（HF）AUROC为0.710，心肌梗死（MI）为0.685，缺血性卒中（IS）为0.655。</p>
  <img src="/images/portfolio/icu-diabetes-cvd-prediction/ROC_Comparison_cvdsubtypes.png" 
       alt="亚型ROC曲线" class="result-img">

  <h3>特征重要性</h3>
  <p>他汀类药物使用、肾病史、硝酸甘油应用、轻度肝病是模型决策核心特征；SHAP分析显示BUN、年龄与CVD风险正相关。</p>
  <img src="/images/portfolio/icu-diabetes-cvd-prediction/shap_summary_plot.png" 
       alt="SHAP summary" class="result-img">

  <!-- 结论与展望 -->
  <div class="conclusion-box">
    <h2 class="section-title" style="margin-top: 0;">结论与展望</h2>
    <ol>
      <li>本项目使用LASSO-XGBoost模型成功预测ICU糖尿病患者的CVD风险，ROC-AUC达到0.713。</li>
      <li>他汀类药物和硝酸甘油使用是ICU糖尿病患者CVD风险的核心因素，需重点临床观察与干预。</li>
      <li>未来可纳入基因、影像等多模态数据，探索深度学习模型进一步提升预测准确性。</li>
    </ol>
  </div>
</div>
