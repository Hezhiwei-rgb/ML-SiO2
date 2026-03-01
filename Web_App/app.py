/* ====== 🚀 单选框 (Radio) 终极核弹级强制拉伸 ====== */
    /* 1. 彻底击穿所有层级的隐藏父容器，强行 100% 宽度 */
    div[data-testid="stRadio"], 
    div[data-testid="stRadio"] > div,
    div[data-testid="stRadio"] > div > div {
        width: 100% !important;
        min-width: 100% !important;
        display: block !important;
    }

    /* 2. 目标实体框：底色、边框、阴影与旁边的输入框保持一致 */
    div[role="radiogroup"] {
        width: 100% !important; 
        min-width: 100% !important;
        height: 100px !important; 
        min-height: 100px !important; 
        border-radius: 15px !important; 
        /* 修改底色：使用稍微偏灰的颜色（与数字输入框的视觉效果匹配）*/
        background-color: #f4f6f9 !important; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.08) !important; 
        border: 2px solid #e0e0e0 !important; 
        display: flex !important; 
        flex-direction: row !important;
        align-items: center !important; 
        justify-content: space-evenly !important; 
        padding: 0 10px !important; 
        box-sizing: border-box !important;
    }

    div[role="radiogroup"]:hover {
        border-color: #4b6cb7 !important; 
        box-shadow: 0 12px 24px rgba(75, 108, 183, 0.2) !important;
    }

    /* 3. 内部选项排版：等分空间 */
    div[role="radiogroup"] label {
        flex: 1 1 0% !important; 
        display: flex !important; 
        justify-content: center !important; 
        align-items: center !important;
        margin: 0 !important; 
        padding: 0 !important;
        cursor: pointer !important; 
    }

    /* 4. 字体大小与颜色：与其他输入框同步 */
    div[role="radiogroup"] label p {
        font-family: 'Arial', sans-serif !important;
        font-size: 2.8rem !important; /* 放大到 2.8rem 匹配其他框，同时防止三个长单词溢出 */
        font-weight: 400 !important; /* 匹配其他输入框的字重 */
        color: #333 !important;
        margin-left: 10px !important;
        white-space: nowrap !important;
    }

    /* 5. 放大单选框前面的小圆圈，使其与大字体更协调 */
    div[role="radiogroup"] div[data-baseweb="radio"] div {
        height: 26px !important;
        width: 26px !important;
    }
