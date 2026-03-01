import streamlit as st
import pandas as pd
import time
import base64
import joblib  
import os
import numpy as np
import json

# ==================== 1. 页面配置 ====================
st.set_page_config(
    page_title="SiO₂ SSA Prediction",
    page_icon="⚗️",
    layout="wide"
)

# ==================== 2. 背景图设置 ====================
def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

bg_path = "Web_App/background.png"
try:
    set_bg_local(bg_path)
except FileNotFoundError:
    st.markdown('<style>.stApp {background-color: #f0f2f6;}</style>', unsafe_allow_html=True)

# ==================== 3. CSS 样式 ====================
st.markdown("""
<style>
    /* === 全局字体 === */
    html, body, [class*="css"] { font-family: 'Arial', sans-serif !important; }

    /* === 容器 === */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
        margin-top: 20px;
        max-width: 1400px;
    }

    /* === 标题 === */
    .main-title {
        font-family: 'Arial', sans-serif !important;
        font-size: 6rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #26557B, #5080A5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 60px;
        filter: drop-shadow(3px 3px 5px rgba(38, 85, 123, 0.2));
    }

    /* === 输入框与标签 (包含了 Radio 的居中大标题) === */
    .stSelectbox label, .stNumberInput label, .stRadio > label {
        display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; margin-bottom: 12px !important;
    }
    .stSelectbox label p, .stNumberInput label p, .stRadio > label p {
        font-family: 'Arial', sans-serif !important; font-size: 3.0rem !important; font-weight: 400 !important; color: #000000 !important; text-align: center !important; 
    }
    div[data-testid="stNumberInput"] > div, div[data-baseweb="select"] > div {
        height: 100px !important; min-height: 100px !important; border-radius: 15px !important; background-color: white !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08) !important; border: 2px solid #e0e0e0 !important; transition: all 0.3s !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
    }
    div[data-testid="stNumberInput"] > div:hover, div[data-baseweb="select"] > div:hover {
        border-color: #4b6cb7 !important; box-shadow: 0 12px 24px rgba(75, 108, 183, 0.2) !important;
    }
    input[type="number"] {
        font-family: 'Arial', sans-serif !important; font-size: 3.2rem !important; font-weight: 400 !important; color: #333 !important; text-align: center !important; min-height: 100px !important; padding: 0 !important;
    }
    div[data-testid="stNumberInput"] button { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }

    /* === 单选框 (Radio) 全横向卡片化样式 === */
    div[role="radiogroup"] {
        height: 100px !important; min-height: 100px !important; border-radius: 15px !important; background-color: white !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08) !important; border: 2px solid #e0e0e0 !important; transition: all 0.3s !important;
        display: flex !important; align-items: center !important; justify-content: space-evenly !important; padding: 0 20px !important;
    }
    div[role="radiogroup"]:hover {
        border-color: #4b6cb7 !important; box-shadow: 0 12px 24px rgba(75, 108, 183, 0.2) !important;
    }
    div[role="radiogroup"] label {
        cursor: pointer !important; margin: 0 !important;
    }
    div[role="radiogroup"] label p {
        font-size: 2.4rem !important; /* 字体放大，撑满全屏的横向空间 */
        font-weight: 600 !important;
        color: #333 !important;
        margin-left: 12px !important;
    }

    /* === 按钮 === */
    div.stButton > button {
        font-family: 'Arial', sans-serif !important; font-size: 3.2rem !important; font-weight: 600 !important; 
        letter-spacing: 2px !important; text-transform: uppercase !important; width: 100% !important; height: 110px !important; margin-top: 20px !important;
        color: #ffffff !important; background-color: #003366 !important; border: none !important; border-radius: 18px !important;
        box-shadow: 0 5px 15px rgba(0, 51, 102, 0.3) !important; transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button p { font-size: 3.2rem !important; font-weight: 600 !important; }
    div.stButton > button:hover { background-color: #002244 !important; transform: translateY(-2px) !important; }
    div.stButton > button:active { background-color: #001122 !important; transform: translateY(2px) !important; }

    /* === 结果框 === */
    .result-container { display: flex; justify-content: center; gap: 50px; margin-top: 50px; padding-bottom: 80px; }
    .result-box {
        background: white; border-top: 10px solid #000000; border-radius: 20px; width: 450px; padding: 40px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    .res-val { font-family: 'Arial', sans-serif !important; color: #000000; font-size: 7rem; font-weight: 800; line-height: 1.1; margin: 10px 0; }
    .res-label { font-family: 'Arial', sans-serif !important; font-size: 2.2rem; font-weight: 700; color: #000000; margin-bottom: 10px; }
    .res-unit { font-family: 'Arial', sans-serif !important; color: #333333; font-size: 2.0rem; font-weight: 400; }
</style>
""", unsafe_allow_html=True)

# ==================== 4. 加载模型与配置 ====================
MODEL_DIR = "Web_App"

models = {
    'base': None, 'preprocessor_x': None, 'cols': None,
    'bt_knn': None, 'bt_scaler_y_base': None, 'bt_scaler_y_target': None,
    'pl_knn': None, 'pl_scaler_y_base': None, 'pl_scaler_y_target': None
}
config = {"BASE_MODEL_WEIGHT_BT": 1.0, "BASE_MODEL_WEIGHT_PL": 1.0}

path_config = os.path.join(MODEL_DIR, "transfer_config.json")
if os.path.exists(path_config):
    with open(path_config, 'r', encoding='utf-8') as f:
        config.update(json.load(f))

# --- 加载基准模型 ---
try:
    models['base'] = joblib.load(os.path.join(MODEL_DIR, "best_ssa_model_base.pkl"))
    models['cols'] = joblib.load(os.path.join(MODEL_DIR, "model_columns.pkl"))
    models['preprocessor_x'] = joblib.load(os.path.join(MODEL_DIR, "preprocessor_x_global.pkl"))
except Exception as e:
    st.error(f"❌ 基准模型加载失败！原因: {e}")

# --- 加载黑滑石 (Black Talc) 迁移组件 ---
try:
    models['bt_knn'] = joblib.load(os.path.join(MODEL_DIR, "transfer_knn_bt.pkl"))
    models['bt_scaler_y_base'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_base_bt.pkl"))
    models['bt_scaler_y_target'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_target_bt.pkl"))
except Exception as e:
    st.sidebar.warning(f"⚠️ Black Talc 迁移模型未加载: {e}")

# --- 加载凹凸棒石 (Attapulgite) 迁移组件 ---
try:
    models
