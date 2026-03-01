import streamlit as st
import pandas as pd
import time
import base64
import joblib  # 👈 核心库
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


bg_path = "background.png"
try:
    set_bg_local(bg_path)
except FileNotFoundError:
    st.markdown('<style>.stApp {background-color: #f0f2f6;}</style>', unsafe_allow_html=True)

# ==================== 3. CSS 样式 (保持你的精美样式) ====================
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

    /* === 输入框与标签 === */
    .stSelectbox label, .stNumberInput label {
        display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; margin-bottom: 12px !important;
    }
    .stSelectbox label p, .stNumberInput label p {
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
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
         display: flex !important; justify-content: center !important; align-items: center !important;
         font-family: 'Arial', sans-serif !important; font-size: 3.2rem !important; font-weight: 400 !important; color: #333 !important; padding-left: 0px !important;
    }
    div[data-testid="stNumberInput"] button { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }

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

# ==================== 4. 智能加载所有模型与配置 ====================
# 设置模型文件夹路径 (请确保路径与训练脚本输出的路径完全一致)
MODEL_DIR = "."

# --- A. 初始化容器 ---
models = {
    'base': None, 'preprocessor_x': None, 'cols': None,
    'bt_knn': None, 'bt_scaler_y_base': None, 'bt_scaler_y_target': None,
    'pl_knn': None, 'pl_scaler_y_base': None, 'pl_scaler_y_target': None
}
config = {"BASE_MODEL_WEIGHT_BT": 1.0, "BASE_MODEL_WEIGHT_PL": 1.0}

# --- B. 加载配置文件 ---
path_config = os.path.join(MODEL_DIR, "transfer_config.json")
try:
    if os.path.exists(path_config):
        with open(path_config, 'r', encoding='utf-8') as f:
            config.update(json.load(f))
except Exception as e:
    st.warning(f"⚠️ 无法加载配置文件，使用默认权重 1.0 ({e})")

# --- C. 加载蒙脱石基准模型 (必需) ---
try:
    models['base'] = joblib.load(os.path.join(MODEL_DIR, "best_ssa_model_base.pkl"))
    models['cols'] = joblib.load(os.path.join(MODEL_DIR, "model_columns.pkl"))
    models['preprocessor_x'] = joblib.load(os.path.join(MODEL_DIR, "preprocessor_x_global.pkl"))
except Exception as e:
    st.error(f"❌ 基准模型加载失败，预测功能不可用！原因: {e}")

# --- D. 加载黑滑石 (Black Talc) 迁移组件 ---
try:
    models['bt_knn'] = joblib.load(os.path.join(MODEL_DIR, "transfer_knn_bt.pkl"))
    models['bt_scaler_y_base'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_base_bt.pkl"))
    models['bt_scaler_y_target'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_target_bt.pkl"))
except Exception:
    pass # 界面上不报错，只在后台静默处理

# --- E. 加载坡缕石/凹凸棒石 (Attapulgite/Palygorskite) 迁移组件 ---
try:
    models['pl_knn'] = joblib.load(os.path.join(MODEL_DIR, "transfer_knn_pl.pkl"))
    models['pl_scaler_y_base'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_base_pl.pkl"))
    models['pl_scaler_y_target'] = joblib.load(os.path.join(MODEL_DIR, "scaler_y_target_pl.pkl"))
except Exception:
    pass
# ==================== 5. 界面布局 ====================
logo_path = "images.png"
col_left, col_center, col_right = st.columns([1, 4, 1])

with col_left:
    try:
        if os.path.exists(logo_path):
            st.image(logo_path, width=230)
    except:
        pass

with col_center:
    st.markdown('<div class="main-title">Porous Nano SiO₂ SSA Prediction</div>', unsafe_allow_html=True)

st.write("")

# --- 输入区 ---
r1_c1, r1_c2 = st.columns(2, gap="large")
with r1_c1:
    clay_options = ["Montmorillonite", "Black Talc", "Sepiolite", "Kaolinite", "Attapulgite"]
    clay_type = st.selectbox("Clay Mineral Type", clay_options)
with r1_c2:
    particle_size = st.number_input("Particle Size (μm)", value=3.0, step=1.0, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

op_c1, op_c2, op_c3 = st.columns(3, gap="large")
with op_c1:
    temp = st.number_input("Temperature (°C)", value=30.0, step=1.0, format="%.0f")
with op_c2:
    sl_ratio = st.number_input("S/L Ratio (g/mL)", value=20.0, step=1.0, format="%.0f")
with op_c3:
    time_val = st.number_input("Reaction Time (h)", value=5.0, step=0.5, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

rm_c1, rm_c2, rm_c3 = st.columns([1, 4, 1])
with rm_c2:
    acid_conc = st.number_input("Acid Conc. (M)", value=2.0, step=0.1, format="%.2f")

# ==================== 6. 核心预测逻辑 (含双重归一化与反演) ====================
st.markdown("<br>", unsafe_allow_html=True)
b_left, b_center, b_right = st.columns([5, 2, 5])

if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'model_source' not in st.session_state:
    st.session_state['model_source'] = ""

with b_center:
    predict_btn = st.button("SSA 🚀", use_container_width=True)

if predict_btn:
    if models['base'] is None or models['cols'] is None:
        st.error("❌ 基准模型未正确加载，无法进行预测。")
    else:
        with st.spinner("Calculating via Transfer Learning..."):
            time.sleep(0.3)

            # 1. 构造输入数据 (使用字典确保映射安全)
            # 注意：键名必须与之前训练时的列名一致！
            data = {
                'SL_Ratio_Num': sl_ratio,
                'Feature1': acid_conc,
                'Feature3': temp,
                'Feature4': time_val,
                'Feature5': particle_size
            }
            input_df = pd.DataFrame(data, index=[0])

            # 2. 强制按照模型训练时的特征顺序重排行
            # 这一步极其重要，防止因为列顺序错乱导致预测出离谱的值
            try:
                input_df = input_df[models['cols']]
            except KeyError as e:
                st.error(f"❌ 输入特征与模型所需的特征不匹配: {e}")
                st.stop()

            try:
                # --- 3. 第一步：获取基准模型的原始预测值 ---
                base_pred_raw = models['base'].predict(input_df)[0]

                final_val = 0.0
                source_msg = ""

                # --- 4. 分支逻辑 ---
                # 情况 A: 黑滑石 (Black Talc)
                if clay_type == "Black Talc" and models['bt_knn'] is not None:
                    # 获取专用组件
                    knn = models['bt_knn']
                    scaler_base = models['bt_scaler_y_base']
                    scaler_target = models['bt_scaler_y_target']
                    weight = config.get("BASE_MODEL_WEIGHT_BT", 1.0)

                    # 归一化基础预测值，并乘以权重
                    # 注意：sklearn 的 transform 要求 2D 数组，所以用 [[ ]]
                    base_scaled = scaler_base.transform([[base_pred_raw]])[0][0] * weight

                    # 获取标准化的输入特征
                    x_scaled = models['preprocessor_x'].transform(input_df)

                    # 预测归一化的残差
                    res_scaled = knn.predict(x_scaled)[0]

                    # 组合后反归一化 (还原为真实 SSA 范围)
                    final_scaled = base_scaled + res_scaled
                    final_val = scaler_target.inverse_transform([[final_scaled]])[0][0]

                    source_msg = "Transfer Learning: KNN Residual Correction (BT)"

                # 情况 B: 坡缕石/凹凸棒石 (Attapulgite/Palygorskite)
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

                # 情况 C: 蒙脱石 (或选了黑滑石/坡缕石但找不到迁移文件)
                else:
                    final_val = base_pred_raw
                    if clay_type in ["Black Talc", "Attapulgite"]:
                        source_msg = f"Warning: {clay_type} transfer files missing. Using Base MMT Model."
                    elif clay_type != "Montmorillonite":
                        source_msg = f"Warning: No transfer model for {clay_type}. Using Base MMT Model."
                    else:
                        source_msg = "Base Model: Montmorillonite (Direct)"

                # 5. 防越界处理 (物理约束：SSA 不可能小于 0)
                if final_val < 0:
                    final_val = 0.0

                # 6. 存储结果渲染
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

    # 动态改变边框颜色 (黑滑石用深灰色，蒙脱石用绿色)
    border_color = "#333333" if clay_type == "Black Talc" else "#28a745"

    st.markdown(f"""
    <div class="result-container">
        <div class="result-box">
            <div class="res-label">Predicted SSA</div>
            <div class="res-val">{res:.1f}</div>
            <div class="res-unit">m²/g</div>
        </div>
        <div class="result-box" style="border-top-color:{border_color};">
             <div class="res-label">Mineral Strategy</div>
             <div class="res-val" style="color:{border_color}; font-size:3.5rem; margin-top:20px;">{clay_type}</div>
             <div class="res-unit" style="font-size:1.2rem; color:#666;">{src}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
