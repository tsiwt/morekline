import os

def write_file(path, content):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip() + "\n")
    print(f"✅ 生成文件: {path}")

def generate_project():
    print("🚀 开始生成 CatBoost 量化预测项目结构...")

    # =========================================================
    # 1. 生成空目录占位符 (确保Git能追踪Models和Predictions)
    # =========================================================
    write_file("Models/.keep", "Keep this directory")
    write_file("Predictions/.keep", "Keep this directory")

    # =========================================================
    # 2. 生成依赖文件 requirements.txt
    # =========================================================
    write_file("requirements.txt", """
pandas
numpy
baostock
catboost
scipy
""")

    # =========================================================
    # 3. 生成核心预测脚本 predict_action.py
    # =========================================================
    predict_code = """
import os
import glob
import pandas as pd
import numpy as np
import baostock as bs
import pickle
from datetime import datetime, timedelta
from scipy.stats import rankdata
from catboost import CatBoostRegressor

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "Models")
PREDS_DIR = os.path.join(SCRIPT_DIR, "Predictions")

if not os.path.exists(PREDS_DIR):
    os.makedirs(PREDS_DIR)

class Global_CatBoost_Manager:
    def __init__(self):
        self.models = {} 
        self.is_trained = False
        self.train_timestamp = ""
        self.model_signature = "Unknown"
        self.feature_names = [
            'board_type', 'price_rank', 'price_d_max', 'price_d_min', 
            'high_rank', 'low_rank', 'open_rank', 'open_gap',                  
            'vol_rank', 'vol_d_max', 'vol_d_min', 'amp_rank', 'amp_d_max', 'amp_d_min',
            'pct_rank', 'pct_d_max', 'pct_d_min', 'entity_rank','entity_d_max','entity_d_min',
            'shadow_up', 'shadow_down', 'ma5_bias', 'bb_pb', 'lag_pct_1'                  
        ]

    def process_features(self, df_raw):
        df = df_raw.copy()
        if 'code' not in df.columns: df['board_type'] = 0
        else:
            conditions = [df['code'].str.startswith('sz.30'), df['code'].str.startswith('sh.68')]
            choices = [1, 2]
            df['board_type'] = np.select(conditions, choices, default=0)

        cols_num = ['open', 'high', 'low', 'close', 'volume']
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df = df.sort_values(['date'])
        df['prev_close'] = df['close'].shift(1)
        df['pct'] = df['close'].pct_change()
        
        hl_range = df['high'] - df['low'] + 1e-6
        df['shadow_up'] = (df['high'] - df[['open', 'close']].max(axis=1)) / hl_range
        df['shadow_down'] = (df[['open', 'close']].min(axis=1) - df['low']) / hl_range
        df['open_gap'] = (df['open'] - df['prev_close']) / (df['prev_close'] + 1e-6)

        ma5 = df['close'].rolling(5).mean()
        df['ma5_bias'] = (df['close'] - ma5) / (ma5 + 1e-6)
        
        mb = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        up = mb + 2 * std
        dn = mb - 2 * std
        df['bb_pb'] = (df['close'] - dn) / (up - dn + 1e-6)
        
        df['lag_pct_1'] = df['pct'].shift(1)
        df['entity'] = (df['close'] - df['open']) / (df['prev_close'] + 1e-6)
        df['amp'] = (df['high'] - df['low']) / (df['prev_close'] + 1e-6)
        
        roll_low = df['low'].rolling(32).min()
        roll_high = df['high'].rolling(32).max()
        df['price_rsv'] = (df['close'] - roll_low) / (roll_high - roll_low + 1e-6)

        window = 32
        def get_rank(series):
            return series.rolling(window).apply(lambda x: rankdata(x, method='ordinal')[-1], raw=True) / window
            
        def get_dist(series, is_max=True):
            def dist_func(x):
                idx = x.argmax() if is_max else x.argmin()
                return ((window - 1) - idx) / (window - 1)
            return series.rolling(window).apply(dist_func, raw=True)

        df['price_rank'] = get_rank(df['price_rsv']) 
        df['high_rank'] = get_rank(df['high']); df['low_rank'] = get_rank(df['low'])         
        df['open_rank'] = get_rank(df['open'])       
        df['price_d_max'] = get_dist(df['high'], True); df['price_d_min'] = get_dist(df['low'], False)
        df['vol_rank'] = get_rank(df['volume'])
        df['vol_d_max'] = get_dist(df['volume'], True); df['vol_d_min'] = get_dist(df['volume'], False)
        df['amp_rank'] = get_rank(df['amp'])
        df['amp_d_max'] = get_dist(df['amp'], True); df['amp_d_min'] = get_dist(df['amp'], False)
        df['pct_rank'] = get_rank(df['pct'])
        df['pct_d_max'] = get_dist(df['pct'], True); df['pct_d_min'] = get_dist(df['pct'], False)
        df['entity_rank'] = get_rank(df['entity'])
        df['entity_d_max'] = get_dist(df['entity'], True); df['entity_d_min'] = get_dist(df['entity'], False)

        return df

    def predict_one(self, df_raw):
        if not self.is_trained or not self.models: return None
        if len(df_raw) < 40: return None
        df_feat = self.process_features(df_raw)
        last_row = df_feat.iloc[[-1]]
        if last_row[self.feature_names].isnull().values.any(): return None
        X_pred = last_row[self.feature_names].values
        preds = {}
        for k in range(1, 6):
            if k in self.models: preds[k] = np.exp(self.models[k].predict(X_pred)[0]) - 1
            else: preds[k] = 0.0
        
        status = {'extra': {'board': int(last_row.iloc[0].get('board_type', 0))}}
        return preds, status

class Stock_Context:
    def __init__(self, code, name):
        self.code = code
        self.name = name
        self.last_date = ""
        self.prev_close = 0.0
        self.preds = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
        self.real_rets = {1:None, 2:None, 3:None, 4:None, 5:None}
        self.ui_status = {}

def get_latest_trade_date():
    today_str = datetime.now().strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(end_date=today_str)
    if rs.error_code != '0': return today_str
    trade_dates = rs.get_data()
    real_trade_dates = trade_dates[trade_dates['is_trading_day'] == '1']
    if real_trade_dates.empty: return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return real_trade_dates['calendar_date'].iloc[-1]

def run_headless_prediction():
    print("=======================================")
    print("   云端 CatBoost 量化预测开始运行")
    print("=======================================")
    models = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if not models:
        print("❌ 未找到模型文件！请将 .pkl 文件放入 Models 文件夹。")
        return
    latest_model = max(models, key=os.path.getctime)
    print(f"▶ 正在加载模型: {latest_model}")

    with open(latest_model, 'rb') as f:
        data = pickle.load(f)
        global_model = data.get('model_manager')
        stock_contexts = data.get('contexts', [])

    if not global_model or not stock_contexts:
        print("❌ 模型文件解析失败或无股票列表。")
        return

    bs.login()
    model_end_date_str = get_latest_trade_date()
    start_date_str = (datetime.strptime(model_end_date_str, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")
    
    valid_predictions = []
    total = len(stock_contexts)
    print(f"▶ 目标预测日期: {model_end_date_str}，共 {total} 只股票。")

    for i, ctx in enumerate(stock_contexts):
        if i % 50 == 0: print(f"  ... 进度: {i}/{total} ...")
        try:
            rs_k = bs.query_history_k_data_plus(ctx.code, "date,open,high,low,close,volume", start_date_str, model_end_date_str, frequency="d", adjustflag="3")
            df = rs_k.get_data()
            if len(df) < 40: continue
            
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
            df['code'] = ctx.code

            res = global_model.predict_one(df)
            if res:
                preds_dict, status = res
                last_row = df.iloc[-1]
                
                row = {
                    'code': ctx.code, 'name': ctx.name, 'date': last_row['date'],
                    'close': last_row['close'],
                    'board_type': status.get('extra', {}).get('board', 0),
                    'signature': global_model.model_signature
                }
                for k in range(1, 6): row[f'pred_{k}d'] = preds_dict.get(k, 0)
                valid_predictions.append(row)
        except Exception as e:
            pass
            
    bs.logout()

    if valid_predictions:
        df_res = pd.DataFrame(valid_predictions)
        df_res = df_res.sort_values(by='pred_5d', ascending=False)
        p_date = df_res.iloc[0]['date']
        safe_sig = getattr(global_model, 'model_signature', 'Unknown').replace('.','').replace(':','')
        fname = f"moreklineinfo_Pred_{p_date}__Using_{safe_sig}.csv"
        fpath = os.path.join(PREDS_DIR, fname)
        df_res.to_csv(fpath, index=False, encoding='utf-8-sig')
        print(f"✅ 预测完成！成功生成: {fpath}")
    else:
        print("⚠️ 未生成任何有效预测。")

if __name__ == "__main__":
    run_headless_prediction()
"""
    write_file("predict_action.py", predict_code)

    # =========================================================
    # 4. 生成 GitHub Actions 自动工作流文件 (.yml)
    # 每天北京时间 18:10 (UTC 10:10) 自动运行
    # =========================================================
    workflow_code = """
name: Daily Stock Prediction

on:
  schedule:
    # 每天 UTC 时间 10:10 运行 (相当于北京时间 18:10，刚好在收盘后)
    - cron: '10 10 * * 1-5'  
  workflow_dispatch: # 允许手动点击按钮触发运行

jobs:
  predict:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Prediction Script
        run: python predict_action.py

      - name: Upload Predictions Artifact
        uses: actions/upload-artifact@v4
        with:
          name: Daily-Predictions-CSV
          path: Predictions/*.csv
          retention-days: 7
"""
    write_file(".github/workflows/daily_predict.yml", workflow_code)
    
    # =========================================================
    # 5. 忽略无用文件
    # =========================================================
    write_file(".gitignore", """
__pycache__/
*.pyc
""")

    print("\n🎉 项目基础文件构建完成！")
    print("▶ 下一步：")
    print("  1. 将你训练好的 .pkl 模型文件放入左侧生成的 Models/ 文件夹中。")
    print("  2. 在终端中依次输入以下命令提交代码:")
    print("     git add .")
    print("     git commit -m 'Initial setup'")
    print("     git push")

if __name__ == "__main__":
    generate_project()
