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

# ==================== 3. CSS 样式 (已优化高度与换行) ====================
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
        font-size: 5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #26557B, #5080A5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 50px;
        filter: drop-shadow(3px 3px 5px rgba(38, 85, 123, 0.2));
    }

    /* === 输入框与标签 (通用设置) === */
    .stSelectbox label, .stNumberInput label, .stRadio label {
        display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; margin-bottom: 8px !important;
    }
    .stSelectbox label p, .stNumberInput label p, .stRadio > label p {
        font-family: 'Arial', sans-serif !important; font-size: 1.8rem !important; font-weight: 600 !important; color: #000000 !important; text-align: center !important; 
    }
/* === 专属修复：矿物类型标题强制居中 === */
    div[data-testid="stRadio"] > label {
        display: flex !important; 
        justify-content: center !important; 
        align-items: center !important; 
        width: 280% !important; 
        margin-bottom: 8px !important;
    }
    div[data-testid="stRadio"] > label p {
        font-family: 'Arial', sans-serif !important; 
        font-size: 1.8rem !important; 
        font-weight: 600 !important; 
        color: #000000 !important; 
        text-align: center !important; 
        width: 100% !important;
    }
    /* === 数字输入框 === */
    div[data-testid="stNumberInput"] > div {
        height: 60px !important; min-height: 60px !important; border-radius: 12px !important; background-color: rgba(255,255,255,0.9) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important; border: 2px solid #e0e0e0 !important; transition: all 0.3s !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
    }
    div[data-testid="stNumberInput"] > div:hover {
        border-color: #4b6cb7 !important; box-shadow: 0 8px 16px rgba(75, 108, 183, 0.2) !important;
    }
    input[type="number"] {
        font-family: 'Arial', sans-serif !important; font-size: 1.8rem !important; font-weight: 400 !important; color: #333 !important; text-align: center !important; min-height: 60px !important; padding: 0 !important;
    }
    div[data-testid="stNumberInput"] button { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }

/* === 矿物选择按钮组 (完全自定义接管) === */
    
    /* 1. 确保最外层容器占满全宽 */
    div[data-testid="stRadio"] {
        width: 100% !important;
    }
    
    /* 2. 把垂直排列强行掰成水平排列，并占满 100% 宽度 */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important; 
        flex-direction: row !important; /* CSS 接管水平布局 */
        width: 250% !important;
        gap: 100px !important;
    }
    
    /* 3. 让三个按钮绝对等分空间 */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1 1 0% !important; /* 强制平分，填满空白 */
        width: 100% !important;
        height: 60px !important;
        background-color: rgba(255,255,255,0.9) !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 12px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important;
        cursor: pointer !important;
        transition: all 0.3s !important;
    }
    
    /* 4. 隐藏自带的丑陋圆圈 */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-of-type {
        display: none !important;
    }
    
    /* 5. 修复文本样式，绝对禁止换行 */
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {
        margin: 0 !important;
        font-family: 'Arial', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #333 !important;
        white-space: nowrap !important; /* 保证 BTlc 不换行 */
    }
    
    /* 6. 悬停与选中状态 */
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: #4b6cb7 !important; 
        transform: translateY(-2px) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #003366 !important; 
        border-color: #003366 !important; 
        box-shadow: 0 5px 15px rgba(0, 51, 102, 0.4) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
    }
    /* === 主按钮 === */
    div.stButton > button {
        font-family: 'Arial', sans-serif !important; font-size: 2.5rem !important; font-weight: 700 !important; 
        letter-spacing: 2px !important; width: 100% !important; height: 80px !important; margin-top: 20px !important;
        color: #ffffff !important; background-color: #003366 !important; border: none !important; border-radius: 15px !important;
        box-shadow: 0 5px 15px rgba(0, 51, 102, 0.3) !important; transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button p { font-size: 2.2rem !important; font-weight: 700 !important; }
    div.stButton > button:hover { background-color: #002244 !important; transform: translateY(-2px) !important; }
    div.stButton > button:active { background-color: #001122 !important; transform: translateY(2px) !important; }

    /* === 结果框 === */
    .result-container { display: flex; justify-content: center; gap: 40px; margin-top: 40px; padding-bottom: 50px; }
    .result-box {
        background: white; border-top: 8px solid #000000; border-radius: 18px; width: 400px; padding: 30px; text-align: center; box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    .res-val { font-family: 'Arial', sans-serif !important; color: #000000; font-size: 5rem; font-weight: 800; line-height: 1.1; margin: 10px 0; }
    .res-label { font-family: 'Arial', sans-serif !important; font-size: 1.8rem; font-weight: 700; color: #000000; margin-bottom: 10px; }
    .res-unit { font-family: 'Arial', sans-serif !important; color: #333333; font-size: 1.5rem; font-weight: 400; }
</style>
""", unsafe_allow_html=True)

# ==================== 4. 智能加载所有模型与配置 ====================
MODEL_DIR = "Web_App"

models = {
    'base': None, 'preprocessor_x': None, 'cols': None,
    'bt_knn': None, 'bt_scaler_y_base': None, 'bt_scaler_y_target': None,
    'pl_knn': None, 'pl_scaler_y_base': None, 'pl_scaler_y_target': None
}
config = {"BASE_MODEL_WEIGHT_BT": 1.0, "BASE_MODEL_WEIGHT_PL": 1.0}

path_config = os.path.join(MODEL_DIR, "transfer_config.json")
try:
    if os.path.exists(path_config):
        with open(path_config, 'r', encoding='utf-8') as f:
            config.update(json.load(f))
except Exception as e:
    st.warning(f"⚠️ 无法加载配置文件，使用默认权重 1.0 ({e})")

try:
    models['base'] = joblib.load(os.path.join(MODEL_DIR, "best_ssa_model_base.pkl"))
    models['cols'] = joblib.load(os.path.join(MODEL_DIR, "model_columns.pkl"))
    models['preprocessor_x'] = joblib.load(os.path.join(MODEL_DIR, "preprocessor_x_global.pkl"))
except Exception as e:
    st.error(f"❌ 基准模型加载失败，预测功能不可用！原因: {e}")

try:
    models['bt_knn'] = joblib.load(os.path.join(MODEL_DIR, "transfer_knn_bt.pkl"))
    models['bt_scaler_y_base'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_base_bt.pkl"))
    models['bt_scaler_y_target'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_target_bt.pkl"))
except Exception:
    pass

try:
    models['pl_knn'] = joblib.load(os.path.join(MODEL_DIR, "transfer_knn_pl.pkl"))
    models['pl_scaler_y_base'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_base_pl.pkl"))
    models['pl_scaler_y_target'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_target_pl.pkl"))
except Exception:
    pass

# ==================== 5. 界面布局 (修复大片留白) ====================
logo_path = "Web_App/images.png"
col_left, col_center, col_right = st.columns([1, 4, 1])

with col_left:
    try:
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)
    except:
        pass

with col_center:
    st.markdown('<div class="main-title">Porous Nano SiO₂ SSA Prediction</div>', unsafe_allow_html=True)

st.write("")

# --- 输入区 (利用空列向中心挤压，缩小两侧留白) ---

# 第一行：把左右留白从 1 缩小到 0.3
spacer1_1, r1_c1, r1_c2, spacer1_2 = st.columns([0.3, 3, 3, 0.3], gap="large")
with r1_c1:
    display_clay_type = st.radio("Clay Mineral Type", ["Mnt", "BTlc", "Pal"])
with r1_c2:
    particle_size = st.number_input("Particle Size (μm)", value=3.00, step=1.0, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

# 第二行：同样缩小左右留白
spacer2_1, op_c1, op_c2, op_c3, spacer2_2 = st.columns([0.3, 2, 2, 2, 0.3], gap="large")
with op_c1:
    temp = st.number_input("Temperature (°C)", value=30.0, step=1.0, format="%.0f")
with op_c2:
    sl_ratio = st.number_input("S/L Ratio (g/mL)", value=20.0, step=1.0, format="%.0f")
with op_c3:
    time_val = st.number_input("Reaction Time (h)", value=5.00, step=0.5, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

# 第三行：调整酸浓度的比例，让它稍微宽一点
spacer3_1, rm_c2, spacer3_2 = st.columns([1.5, 4, 1.5])
with rm_c2:
    acid_conc = st.number_input("Acid Conc. (M)", value=2.00, step=0.1, format="%.2f")

# ==================== 6. 核心预测逻辑 ====================
st.markdown("<br>", unsafe_allow_html=True)
b_left, b_center, b_right = st.columns([4, 2, 4])

if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'model_source' not in st.session_state:
    st.session_state['model_source'] = ""

with b_center:
    predict_btn = st.button("SSA 🚀", use_container_width=True)

if predict_btn:

    # 将前端显示的缩写映射回后端逻辑需要的全称
    clay_mapping = {
        "Mnt": "Montmorillonite",
        "BTlc": "Black Talc",
        "Pal": "Attapulgite"
    }
    clay_type = clay_mapping[display_clay_type]

    if models['base'] is None or models['cols'] is None:
        st.error("❌ 基准模型未正确加载，无法进行预测。")
    else:
        with st.spinner("Calculating via Transfer Learning..."):
            time.sleep(0.3)

            data = {
                'SL_Ratio_Num': sl_ratio,
                'Feature1': acid_conc,
                'Feature3': temp,
                'Feature4': time_val,
                'Feature5': particle_size
            }
            input_df = pd.DataFrame(data, index=[0])

            try:
                input_df = input_df[models['cols']]
            except KeyError as e:
                st.error(f"❌ 输入特征与模型所需的特征不匹配: {e}")
                st.stop()

            try:
                base_pred_raw = models['base'].predict(input_df)[0]
                final_val = 0.0
                source_msg = ""

                if clay_type == "Black Talc" and models['bt_knn'] is not None:
                    knn = models['bt_knn']
                    scaler_base = models['bt_scaler_y_base']
                    scaler_target = models['bt_scaler_y_target']
                    weight = config.get("BASE_MODEL_WEIGHT_BT", 1.0)

                    base_scaled = scaler_base.transform([[base_pred_raw]])[0][0] * weight
                    x_scaled = models['preprocessor_x'].transform(input_df)
                    res_scaled = knn.predict(x_scaled)[0]

                    final_scaled = base_scaled + res_scaled
                    final_val = scaler_target.inverse_transform([[final_scaled]])[0][0]
                    source_msg = "Transfer Learning: KNN Residual Correction (BT)"

                elif clay_type == "Attapulgite" and models['pl_knn'] is not None:
                    knn = models['pl_knn']
                    scaler_base = models['pl_scaler_y_base']
                    scaler_target = models['pl_scaler_y_target']
                    weight = config.get("BASE_MODEL_WEIGHT_PL", 1.0)

                    base_scaled = scaler_base.transform([[base_pred_raw]])[0][0] * weight
                    x_scaled = models['preprocessor_x'].transform(input_df)
                    res_scaled = knn.predict(x_scaled)[0]

                    final_scaled = base_scaled + res_scaled
                    final_val = scaler_target.inverse_transform([[final_scaled]])[0][0]
                    source_msg = "Transfer Learning: KNN Residual Correction (PL)"

                else:
                    final_val = base_pred_raw
                    if clay_type in ["Black Talc", "Attapulgite"]:
                        source_msg = f"Warning: {clay_type} transfer files missing. Using Base MMT Model."
                    else:
                        source_msg = "Base Model: Montmorillonite (Direct)"

                if final_val < 0:
                    final_val = 0.0

                st.session_state['prediction_result'] = final_val
                st.session_state['model_source'] = source_msg

            except Exception as e:
                st.error(f"计算过程中发生错误: {e}")
                import traceback

                st.error(traceback.format_exc())

# ==================== 7. 结果显示 ====================
if st.session_state['prediction_result'] is not None:
    res = st.session_state['prediction_result']
    src = st.session_state['model_source']

    border_color = "#333333" if display_clay_type == "BTlc" else "#28a745"

    st.markdown(f"""
    <div class="result-container">
        <div class="result-box">
            <div class="res-label">Predicted SSA</div>
            <div class="res-val">{res:.1f}</div>
            <div class="res-unit">m²/g</div>
        </div>
        <div class="result-box" style="border-top-color:{border_color};">
             <div class="res-label">Mineral Strategy</div>
             <div class="res-val" style="color:{border_color}; font-size:3rem; margin-top:20px;">{display_clay_type}</div>
             <div class="res-unit" style="font-size:1.1rem; color:#666;">{src}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)









