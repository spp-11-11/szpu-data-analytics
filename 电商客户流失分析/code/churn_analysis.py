# -*- coding: utf-8 -*-
"""
电商客户流失分析 —— 依照《心脏病发病预测实训报告书》流程完成
流程：数据探查与清洗 → EDA 可视化 → 多模型预测对比 → 算法优化 → 结论与预测输出

说明：
- 原始数据在 rawData.csv，共 40450 行 19 列；R 列（第 18 列，索引 17）Churned 为目标变量，
  1=流失，0=留存。
- 最终输出 prediction_result.csv，仅两列：行号（从 0 开始）、预测是否流失（0/1）。
- 所有代码均带中文注释，解释"做什么、为什么这么做"。
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')

# ---------- 可复现与中文显示 ----------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_theme(style='whitegrid', palette='Set2')  # 先设主题，避免其重置字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 的上一级 = 项目根目录
FIG_DIR = os.path.join(BASE_DIR, 'img')
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = 'Churned'
CAT_COLS = ['Gender', 'Country', 'City', 'Signup_Quarter']
NUM_COLS = [
    'Age', 'Session_Duration_Avg', 'Pages_Per_Session', 'Total_Purchases',
    'Average_Order_Value', 'Days_Since_Last_Purchase', 'Discount_Usage_Rate',
    'Returns_Rate', 'Email_Open_Rate', 'Customer_Service_Calls',
    'Product_Reviews_Written', 'Payment_Method_Diversity', 'Lifetime_Value',
    'Credit_Balance',
]


def load_raw():
    """读取原始数据。"""
    path = os.path.join(BASE_DIR, 'data', 'rawData.csv')
    df = pd.read_csv(path)
    print('数据规模：', df.shape, '（行数 × 列数）')
    print('目标列位置：第', df.columns.get_loc(TARGET) + 1, '列（Excel 列 R）')
    return df


def profile(df):
    """1.1~1.4 数据探查：类型、缺失、重复、异常值、类别取值。"""
    print('\n=== 字段类型 ===')
    print(df.dtypes.to_string())
    print('\n=== 缺失值统计 ===')
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print('\n=== 重复行数 ===', df.duplicated().sum())
    print('\n=== 目标变量分布 ===')
    print(df[TARGET].value_counts().to_string())
    print('流失率：%.2f%%' % (df[TARGET].mean() * 100))
    print('\n=== 类别字段原始取值 ===')
    for col in CAT_COLS:
        print(col, '类别数 =', df[col].nunique(), '示例 =', df[col].value_counts().head(6).to_dict())


def clean_data(df):
    """
    1.5 执行清洗（返回一个新的 DataFrame，不改变原始行索引）。

    清洗策略：
    1) Total_Purchases 中的 '13.0次'、Average_Order_Value 中的 '$159.94'
       → 用正则提取数字，转回数值型（这是脏数据，不是普通字符串）。
    2) 类别字段规范化：Country/City 去首尾空格并统一大小写写法；
       Gender 的 'F' 归一为 'Female'，'1' 归一为 'Male'（字典只允许三类）。
    3) 违反字典范围的数值视为异常 → 置为 NaN → 用中位数填充（比均值稳健）。
    4) 其余缺失值用中位数填充。
    5) 不在本函数去重：因为最终要为原始 40450 行全部输出预测，
       去重只在建模评估阶段单独进行，避免丢失行号对应关系。
    """
    data = df.copy()

    # (1) 带单位的字符串 → 数值
    for col in ['Total_Purchases', 'Average_Order_Value']:
        data[col] = pd.to_numeric(
            data[col].astype(str).str.extract(r'([-+]?\d+\.?\d*)')[0], errors='coerce')

    # (2) 类别规范化
    data['Gender'] = data['Gender'].astype(str).str.strip().replace(
        {'F': 'Female', '1': 'Male'})
    # Country 字典只有 8 类，把大小写/空格变体统一到标准写法
    canon_country = {'usa': 'USA', 'uk': 'UK', 'canada': 'Canada', 'germany': 'Germany',
                     'australia': 'Australia', 'france': 'France', 'india': 'India',
                     'japan': 'Japan'}
    data['Country'] = data['Country'].astype(str).str.strip().str.lower().map(
        lambda x: canon_country.get(x, x.title()))
    data['City'] = data['City'].astype(str).str.strip()
    data['Signup_Quarter'] = data['Signup_Quarter'].astype(str).str.strip()

    # (3) 违反字典范围的数值 → NaN
    # 字典：Age 0~65；会话时长应 >= 0；比率类 0~100；退货率 0~100
    data.loc[(data['Age'] < 0) | (data['Age'] > 65), 'Age'] = np.nan
    data.loc[data['Session_Duration_Avg'] < 0, 'Session_Duration_Avg'] = np.nan
    for col in ['Discount_Usage_Rate', 'Email_Open_Rate', 'Returns_Rate']:
        data.loc[(data[col] < 0) | (data[col] > 100), col] = np.nan
    # Credit_Balance = 0 在字典中未标为异常，保留（可能确实无余额）
    # Lifetime_Value = 0、Pages_Per_Session = 0 同理保留

    # (4) 中位数填充所有数值列的缺失
    fill_log = {}
    for col in NUM_COLS:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            med = data[col].median()
            data[col] = data[col].fillna(med)
            fill_log[col] = (n_missing, round(med, 2))
    print('\n=== 中位数填充记录（字段: 填充个数, 中位数）===')
    print(fill_log)
    print('清洗后缺失值总数：', data.isnull().sum().sum())
    return data


def make_features(data):
    """
    3.1 特征工程：
    - 无序多类别（Gender/Country/City/Signup_Quarter）用独热编码 drop_first=True 避免虚拟变量陷阱；
    - 数值特征保持原样，标准化放进 Pipeline 里做，防止数据泄漏。
    """
    X = pd.get_dummies(data.drop(columns=[TARGET]), columns=CAT_COLS, drop_first=True).astype(float)
    y = data[TARGET].astype(int)
    return X, y


def eda(data):
    """2. EDA 可视化：目标分布、数值分布、类别流失率、相关性、重点特征。"""
    # 2.1 目标分布
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cnt = data[TARGET].map({0: '留存', 1: '流失'}).value_counts()
    sns.countplot(x=data[TARGET].map({0: '留存', 1: '流失'}), ax=axes[0], order=['留存', '流失'])
    axes[0].set_title('目标变量分布（计数）'); axes[0].set_xlabel('是否流失'); axes[0].set_ylabel('样本数')
    for i, v in enumerate(cnt):
        axes[0].text(i, v + 100, str(v), ha='center')
    axes[1].pie(cnt.values, labels=cnt.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('目标变量分布（占比）')
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, '01_目标分布.png'), dpi=110); plt.close()

    # 2.2 数值特征分布（按是否流失对比），选取有代表性的 6 个
    show_num = ['Age', 'Session_Duration_Avg', 'Total_Purchases', 'Days_Since_Last_Purchase',
                'Lifetime_Value', 'Customer_Service_Calls']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flat, show_num):
        for label, name in [(0, '留存'), (1, '流失')]:
            sns.histplot(data[data[TARGET] == label][col], ax=ax, label=name,
                         kde=True, stat='density', alpha=0.5)
        ax.set_title(f'{col} 分布对比'); ax.set_xlabel(col); ax.set_ylabel('密度'); ax.legend()
    plt.suptitle('数值特征在两类客户中的分布', y=1.01, fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, '02_数值特征分布.png'), dpi=110, bbox_inches='tight'); plt.close()

    # 2.3 类别特征与流失率关系
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col in zip(axes.flat, CAT_COLS):
        if col == 'City':
            # 城市 40 类太多，只显示流失率最高/最低的各 5 个（且样本数 > 200，避免小样本噪声）
            g = data.groupby(col)[TARGET].agg(['mean', 'count'])
            g = g[g['count'] > 200].sort_values('mean')
            rate = pd.concat([g.head(5), g.tail(5)])['mean'] * 100
        else:
            rate = data.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, palette='Set2')
        ax.set_title(f'不同 {col} 的流失率'); ax.set_xlabel(col); ax.set_ylabel('流失率 (%)')
        ax.set_ylim(0, 100); ax.tick_params(axis='x', rotation=35)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, '03_类别特征流失率.png'), dpi=110); plt.close()

    # 2.4 相关性热力图
    corr_cols = NUM_COLS + [TARGET]
    corr = data[corr_cols].corr()
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=.3, cbar_kws={'label': '皮尔逊相关系数'})
    plt.title('数值特征相关性热力图'); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '04_相关性热力图.png'), dpi=110); plt.close()
    print('\n=== 与目标相关性最强的数值特征 ===')
    print(corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).round(3).to_string())

    # 2.5 重点特征深挖（类比任务书点名的 Oldpeak/MaxHR）
    focus = ['Days_Since_Last_Purchase', 'Total_Purchases']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, focus):
        sns.boxplot(x=data[TARGET].map({0: '留存', 1: '流失'}), y=data[col], ax=ax, palette='Set2')
        ax.set_title(f'{col} 在两类客户中的分布'); ax.set_xlabel('是否流失'); ax.set_ylabel(col)
    plt.suptitle('重点特征对比', y=1.02, fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, '05_重点特征对比.png'), dpi=110); plt.close()
    print('\n=== 重点特征分组均值 ===')
    print(data.groupby(TARGET)[focus].mean().round(2).to_string())


def build_models():
    """3.2 定义 6 种模型（≥4 即可，这里做 6 种便于横向对比）。"""
    return {
        '逻辑回归': Pipeline([('scaler', StandardScaler()),
                          ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]),
        '决策树': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
        '随机森林': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        '支持向量机': Pipeline([('scaler', StandardScaler()),
                          ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))]),
        'KNN': Pipeline([('scaler', StandardScaler()),
                        ('clf', KNeighborsClassifier(n_neighbors=5))]),
        '梯度提升': GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def main():
    # ===== 0. 读取与探查 =====
    raw = load_raw()
    profile(raw)

    # ===== 1. 数据清洗 =====
    data = clean_data(raw)
    # 清洗后复核：目标分布
    print('\n清洗后目标分布：')
    print(data[TARGET].value_counts().to_string())

    # ===== 2. EDA =====
    eda(data)

    # ===== 3. 多模型对比（建模数据去重，避免同一条样本同时出现在训练/测试集）=====
    n_raw = len(data)
    model_data = data.drop_duplicates().reset_index(drop=True)
    print('\n建模数据：原始 %d 行 → 去重后 %d 行（去重 %d 行）' % (n_raw, len(model_data), n_raw - len(model_data)))
    X, y = make_features(model_data)
    print('特征维度：', X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {'准确率': 'accuracy', '精确率': 'precision', '召回率': 'recall',
               'F1': 'f1', 'AUC': 'roc_auc'}
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({'模型': name,
                     '准确率': res['test_准确率'].mean(),
                     '精确率': res['test_精确率'].mean(),
                     '召回率': res['test_召回率'].mean(),
                     'F1': res['test_F1'].mean(),
                     'AUC': res['test_AUC'].mean()})
    cv_results = pd.DataFrame(rows).set_index('模型').round(4).sort_values('AUC', ascending=False)
    print('\n=== 5 折交叉验证结果（训练集均值）===')
    print(cv_results.to_string())

    ax = cv_results.plot(kind='bar', figsize=(12, 5), width=0.8)
    ax.set_title('多模型交叉验证指标对比'); ax.set_xlabel('模型'); ax.set_ylabel('得分')
    ax.set_ylim(0, 1.05); ax.legend(loc='lower right'); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, '06_多模型对比.png'), dpi=110); plt.close()

    # ===== 4. 算法优化 =====
    # 方法 1：对 AUC 最高的模型做网格搜索
    best_name = cv_results.index[0]
    print('\n交叉验证 AUC 最优模型：', best_name)
    param_grids = {
        '逻辑回归': {'clf__C': [0.01, 0.1, 1, 10, 100]},
        '决策树': {'max_depth': [4, 6, 8, 10, 12], 'min_samples_leaf': [1, 2, 4, 8]},
        '随机森林': {'n_estimators': [100, 200, 300], 'max_depth': [None, 8, 12],
                 'min_samples_leaf': [1, 2, 4]},
        '支持向量机': {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 0.01, 0.1]},
        'KNN': {'clf__n_neighbors': [3, 5, 7, 9, 11]},
        '梯度提升': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                 'max_depth': [2, 3, 4]},
    }
    grid = GridSearchCV(models[best_name], param_grids[best_name],
                        scoring='roc_auc', cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print('网格搜索最优参数：', grid.best_params_)
    print('网格搜索最优 AUC：%.4f' % grid.best_score_)
    best_model = grid.best_estimator_

    # 方法 2：随机搜索（对随机森林）作补充对比
    rand_dist = {'n_estimators': [100, 200, 300], 'max_depth': [None, 6, 10, 14],
                 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
                 'max_features': ['sqrt', 'log2', None]}
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
                              rand_dist, n_iter=12, scoring='roc_auc', cv=cv,
                              n_jobs=-1, random_state=RANDOM_SEED)
    rand.fit(X_train, y_train)
    print('随机搜索最优参数：', rand.best_params_)
    print('随机搜索最优 AUC：%.4f' % rand.best_score_)

    # 测试集最终评估
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    print('\n=== 测试集分类报告 ===')
    print(classification_report(y_test, y_pred, target_names=['留存', '流失']))
    print('测试集 AUC：%.4f' % roc_auc_score(y_test, y_prob))
    print(pd.Series({'准确率': accuracy_score(y_test, y_pred),
                     '精确率': precision_score(y_test, y_pred),
                     '召回率': recall_score(y_test, y_pred),
                     'F1': f1_score(y_test, y_pred),
                     'AUC': roc_auc_score(y_test, y_prob)}, name=best_name).round(4).to_string())

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=['留存', '流失']).plot(
        ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title('混淆矩阵 — %s（测试集）' % best_name)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, '07_混淆矩阵.png'), dpi=110); plt.close()
    print('混淆矩阵 TN=%d FP=%d FN=%d TP=%d' % tuple(cm.ravel()))

    # ROC 曲线（多模型叠加）
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, lw=1.6, label='%s (AUC=%.3f)' % (name, auc(fpr, tpr)))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测')
    ax.set_title('多模型 ROC 曲线对比（测试集）'); ax.set_xlabel('假阳性率 FPR'); ax.set_ylabel('真阳性率 TPR')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, '08_ROC曲线.png'), dpi=110); plt.close()

    # 排列重要性
    result = permutation_importance(best_model, X_test, y_test, scoring='roc_auc',
                                    n_repeats=8, random_state=RANDOM_SEED, n_jobs=-1)
    imp = pd.DataFrame({'feature': X.columns, 'mean': result.importances_mean,
                        'std': result.importances_std}).sort_values('mean')
    plt.figure(figsize=(8, 7))
    plt.barh(imp['feature'], imp['mean'], xerr=np.asarray(imp['std']), color='#5ab4ac')
    plt.title('特征排列重要性 — %s' % best_name)
    plt.xlabel('AUC 下降幅度'); plt.ylabel('特征'); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '09_特征重要性.png'), dpi=110); plt.close()
    print('\n=== 重要性 Top 10 ===')
    print(imp.sort_values('mean', ascending=False).head(10)[['feature', 'mean']].round(4).to_string(index=False))

    # ===== 5. 结论与最终预测输出 =====
    # 用全部去重后的建模数据重新训练最优参数模型，再对清洗后的全部原始行（含重复行）预测，
    # 保证输出行数 = 原始 40450 行，且行号与 rawData.csv 一一对应（0 开始）。
    final_model = grid.best_estimator_.fit(X, y)
    X_all, _ = make_features(data)
    X_all = X_all.reindex(columns=X.columns, fill_value=0)  # 对齐列，防止列顺序不一致
    pred_all = final_model.predict(X_all).astype(int)

    result = pd.DataFrame({'行号': np.arange(len(data)), '预测是否流失': pred_all})
    out_path = os.path.join(BASE_DIR, '产出', 'prediction_result.csv')
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    print('\n=== 预测结果 ===')
    print(result.head(10).to_string(index=False))
    print('输出文件：', out_path)
    print('输出行数：', len(result), ' 预测流失数：', int(pred_all.sum()),
          ' 预测流失率：%.2f%%' % (pred_all.mean() * 100))


if __name__ == '__main__':
    main()
