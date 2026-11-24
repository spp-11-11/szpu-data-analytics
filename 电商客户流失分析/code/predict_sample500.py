# -*- coding: utf-8 -*-
"""
训练模型 -> 从验证集随机抽取 500 行 -> 预测 -> 输出 prediction_result.csv

rawData.csv           训练集（含 Churned 目标列）
验证集1-发学生.csv      验证集（5000 行，无目标列，待预测）
prediction_result.csv  两列：行号（验证集原始行号）、预测是否流失（0/1），共 500 行

随机种子固定为 42，保证抽取结果可复现。
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from churn_analysis import clean_data, make_features, BASE_DIR, RANDOM_SEED

SAMPLE_SIZE = 500
CAT_COLS = ['Gender', 'Country', 'City', 'Signup_Quarter']


def features_no_target(data):
    """验证集无 Churned 列，直接对特征做独热编码（带 errors='ignore' 更稳妥）。"""
    return pd.get_dummies(data.drop(columns=['Churned'], errors='ignore'),
                          columns=CAT_COLS, drop_first=True).astype(float)


def main():
    # 1) 训练集：清洗 + 去重 + 特征工程，训练网格搜索确定的最优参数模型
    train = clean_data(pd.read_csv(os.path.join(BASE_DIR, 'data', 'rawData.csv')))
    train = train.drop_duplicates().reset_index(drop=True)
    X_train, y_train = make_features(train)
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=4,
                                       n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    print('训练集：%d 行（去重后），%d 特征' % (len(X_train), X_train.shape[1]))

    # 2) 验证集：先整体清洗，再用固定种子随机抽取 500 行
    val = clean_data(pd.read_csv(os.path.join(BASE_DIR, 'data', '验证集1-发学生.csv')))
    rng = np.random.RandomState(RANDOM_SEED)
    picked = np.sort(rng.choice(len(val), size=SAMPLE_SIZE, replace=False))  # 验证集中的原始行号
    val_sample = val.iloc[picked]

    # 3) 特征工程 + 列对齐 + 预测
    X_val = features_no_target(val_sample)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    pred = model.predict(X_val).astype(int)

    # 4) 输出：行号用验证集原始行号，便于核对
    result = pd.DataFrame({'行号': picked, '预测是否流失': pred})
    out = os.path.join(BASE_DIR, '产出', 'prediction_result.csv')
    result.to_csv(out, index=False, encoding='utf-8-sig')
    print('验证集：%d 行，随机抽取 %d 行，种子 %d' % (len(val), SAMPLE_SIZE, RANDOM_SEED))
    print('输出：', out)
    print('行数：', len(result), '| 行号范围：', result['行号'].min(), '-', result['行号'].max())
    print('预测流失数：', int(pred.sum()), '| 预测流失率：%.2f%%' % (pred.mean() * 100))
    print(result.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
