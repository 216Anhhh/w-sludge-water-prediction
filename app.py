# ===== Fix for Streamlit Cloud =====
import matplotlib
matplotlib.use('Agg')
# ==================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import io
warnings.filterwarnings('ignore')

# ===== 中文字体配置 =====
from matplotlib import font_manager

chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STHeiti', 'Heiti SC']
font_set = False
for font_name in chinese_fonts:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        font_set = True
        break
    except:
        continue

if not font_set:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# ============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import shap

# Page config
st.set_page_config(
    page_title="污水处理智能分析平台",
    page_icon="💧",
    layout="wide"
)

# ============ 初始化session_state ============
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'default'
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'pred_values' not in st.session_state:
    st.session_state.pred_values = {}
if 'input_values' not in st.session_state:
    st.session_state.input_values = {}
if 'models' not in st.session_state:
    st.session_state.models = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# ============ 主题配色 ============
def get_theme_colors(theme):
    if theme == 'dark':
        return {
            'bg': '#0e1117', 'bg2': '#0d1117', 'sidebar_bg': '#0d1117',
            'text': '#f0f6fc', 'text_secondary': '#8b949e', 'border': '#30363d',
            'primary': '#58a6ff', 'success': '#3fb950', 'warning': '#d29922', 'danger': '#f85149',
            'card_bg': '#161b22', 'plot_bg': '#0d1117', 'plot_face': '#0d1117',
            'button_bg': '#238636', 'button_hover': '#2ea043', 'button_text': '#ffffff',
            'tab_bg': '#161b22', 'tab_active': '#238636', 'tab_text': '#8b949e', 'tab_active_text': '#ffffff',
            'text_color': '#ffffff', 'input_bg': '#1a1a2e', 'input_text': '#f0f6fc',
            'select_bg': '#1a1a2e', 'select_text': '#f0f6fc',
            'plot_facecolor': '#0d1117', 'plot_textcolor': 'white',
        }
    else:
        return {
            'bg': '#f5f7fa', 'bg2': '#ffffff', 'sidebar_bg': '#e8ecf1',
            'text': '#1a1a2e', 'text_secondary': '#3a4a5a', 'border': '#d0d7de',
            'primary': '#1a5276', 'success': '#1a8a4a', 'warning': '#b87a0a', 'danger': '#b02a37',
            'card_bg': '#ffffff', 'plot_bg': '#ffffff', 'plot_face': '#ffffff',
            'button_bg': '#e8d5b8', 'button_hover': '#dcc4a0', 'button_text': '#1a1a2e',
            'tab_bg': '#ffffff', 'tab_active': '#b8d4e3', 'tab_text': '#4a5a6a', 'tab_active_text': '#1a1a2e',
            'text_color': '#1a1a2e', 'input_bg': '#f0f2f6', 'input_text': '#1a1a2e',
            'select_bg': '#ffffff', 'select_text': '#1a1a2e',
            'plot_facecolor': '#ffffff', 'plot_textcolor': '#1a1a2e',
        }

colors = get_theme_colors(st.session_state.theme)

# ============ 更新matplotlib颜色 ============
def update_matplotlib_theme(theme, colors):
    if theme == 'dark':
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#30363d'
        plt.rcParams['figure.facecolor'] = '#0d1117'
        plt.rcParams['axes.facecolor'] = '#0d1117'
    else:
        plt.rcParams['text.color'] = '#1a1a2e'
        plt.rcParams['axes.labelcolor'] = '#1a1a2e'
        plt.rcParams['xtick.color'] = '#1a1a2e'
        plt.rcParams['ytick.color'] = '#1a1a2e'
        plt.rcParams['axes.edgecolor'] = '#d0d7de'
        plt.rcParams['figure.facecolor'] = '#ffffff'
        plt.rcParams['axes.facecolor'] = '#ffffff'

update_matplotlib_theme(st.session_state.theme, colors)

# ============ CSS ============
def get_css(colors, theme):
    light_overrides = ""
    if theme == 'light':
        light_overrides = """
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        .stMarkdown label, .stMarkdown .label, .stMarkdown .value,
        .result-card .label, .result-card .value,
        .metric-card .label, .metric-card .value, .metric-card .sub,
        div, p, span, label { color: #1a1a2e !important; }
        .result-card, .result-card * { color: #1a1a2e !important; }
        .metric-card, .metric-card * { color: #1a1a2e !important; }
        .status-normal { color: #1a8a4a !important; font-weight: 700; }
        .status-warning { color: #b87a0a !important; font-weight: 700; }
        .status-danger { color: #b02a37 !important; font-weight: 700; }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
            color: #1a1a2e !important;
        }
        .stSelectbox div[data-baseweb="select"] div {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
        }
        .stSelectbox ul { background-color: #ffffff !important; }
        .stSelectbox li { color: #1a1a2e !important; background-color: #ffffff !important; }
        .stSelectbox li:hover { background-color: #e8ecf1 !important; }
        .stNumberInput input, .stTextInput input {
            background-color: #f0f2f6 !important;
            color: #1a1a2e !important;
        }
        """
    
    if theme == 'dark':
        button_css = """
        .stButton button {
            background: #238636; color: #ffffff; font-weight: 700; border: none;
            border-radius: 8px; padding: 0.6rem 2rem; width: 100%;
            transition: all 0.3s ease; font-size: 1rem;
        }
        .stButton button:hover {
            background: #2ea043; transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(35, 134, 54, 0.4);
        }
        """
    else:
        button_css = """
        .stButton button {
            background: #e8d5b8; color: #1a1a2e !important; font-weight: 700;
            border: 2px solid #d4a574; border-radius: 8px; padding: 0.6rem 2rem;
            width: 100%; transition: all 0.3s ease; font-size: 1rem;
        }
        .stButton button:hover {
            background: #dcc4a0; transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(212, 165, 116, 0.4);
        }
        """
    
    return f"""
    <style>
    .stApp {{ background-color: {colors['bg']}; }}
    section[data-testid="stSidebar"] {{
        background-color: {colors['sidebar_bg']} !important;
        border-right: 1px solid {colors['border']} !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown {{ color: {colors['text']} !important; }}
    section[data-testid="stSidebar"] .stMarkdown p {{ color: {colors['text']} !important; }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {{
        color: {colors['text']} !important;
    }}
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox select,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {{
        background-color: {colors['input_bg']} !important;
        color: {colors['input_text']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stSelectbox label {{
        color: {colors['text_secondary']} !important;
    }}
    .stSelectbox div[data-baseweb="select"] div {{
        background-color: {colors['select_bg']} !important;
        color: {colors['select_text']} !important;
    }}
    .stSelectbox ul {{ background-color: {colors['select_bg']} !important; }}
    .stSelectbox li {{ color: {colors['select_text']} !important; background-color: {colors['select_bg']} !important; }}
    .stSelectbox li:hover {{ background-color: {colors['button_hover']} !important; }}
    .main-header {{
        font-size: 2.5rem; font-weight: 700; color: {colors['primary']};
        text-align: center; padding: 1rem 0 0.2rem 0; letter-spacing: 2px;
    }}
    .sub-header {{
        font-size: 1rem; color: {colors['text_secondary']}; text-align: center;
        padding-bottom: 1rem; border-bottom: 1px solid {colors['border']};
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: {colors['card_bg']}; border-radius: 10px; padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,{0.4 if theme=='dark' else 0.08});
        border-left: 4px solid {colors['primary']}; text-align: center; margin: 0 4px;
    }}
    .metric-card .label {{
        font-size: 0.75rem; color: {colors['text_secondary']}; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .metric-card .value {{
        font-size: 1.8rem; font-weight: 700; color: {colors['text']}; margin: 4px 0;
    }}
    .metric-card .sub {{ font-size: 0.7rem; color: {colors['text_secondary']}; }}
    .result-card {{
        background: {colors['card_bg']}; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,{0.4 if theme=='dark' else 0.08});
        text-align: center; border-top: 4px solid {colors['primary']}; height: 100%;
    }}
    .result-card .label {{ font-size: 0.8rem; color: {colors['text_secondary']}; font-weight: 500; }}
    .result-card .value {{ font-size: 2.2rem; font-weight: 700; color: {colors['text']}; margin: 6px 0; }}
    {button_css}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: {colors['tab_bg']}; padding: 6px;
        border-radius: 12px; border: 1px solid {colors['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px; padding: 8px 20px; font-weight: 600;
        color: {colors['tab_text']}; transition: all 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: {colors['tab_active']}; color: {colors['tab_active_text']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .status-normal {{ color: {colors['success']}; font-weight: 700; }}
    .status-warning {{ color: {colors['warning']}; font-weight: 700; }}
    .status-danger {{ color: {colors['danger']}; font-weight: 700; }}
    hr {{ border-color: {colors['border']} !important; }}
    .stAlert {{ background-color: {colors['card_bg']} !important; border-color: {colors['border']} !important; color: {colors['text']} !important; }}
    {light_overrides}
    </style>
    """

st.markdown(get_css(colors, st.session_state.theme), unsafe_allow_html=True)

st.markdown(f'<div class="main-header">💧 污水处理智能分析平台</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">基于进水参数的污泥指标预测与SRT优化系统</div>', unsafe_allow_html=True)

# ============ 加载数据 ============
@st.cache_data
def load_default_data():
    try:
        df = pd.read_excel('随机森林归一化.xlsx', sheet_name='Sheet1')
        return df
    except:
        try:
            df = pd.read_excel('data/随机森林归一化.xlsx', sheet_name='Sheet1')
            return df
        except:
            return None

def load_data():
    if st.session_state.data_source == 'uploaded' and st.session_state.df_loaded is not None:
        return st.session_state.df_loaded
    else:
        return load_default_data()

df = load_data()
if df is None:
    st.error("❌ 找不到数据文件！")
    st.stop()

X_columns = ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)', 'SS(mg/l)', 
             'NH3-N(mg/l)', 'TP(mg/l)', 'TN(mg/l)', 'Tin℃']
y_columns = ['F/M(%)', 'SVI', 'SRT']

x_names_cn = {
    'Qoutm3/d': '进水流量', 'BOD5 (mg/l)': '进水BOD5',
    'CODcr(mg/l)': '进水CODcr', 'SS(mg/l)': '进水SS',
    'NH3-N(mg/l)': '进水NH3-N', 'TP(mg/l)': '进水TP',
    'TN(mg/l)': '进水TN', 'Tin℃': '进水水温'
}
y_names_cn = {
    'F/M(%)': '有机质占比',
    'SVI': 'SVI (污泥体积指数)',
    'SRT': 'SRT (污泥龄)'
}
x_names_en = {
    'Qoutm3/d': 'Flow Rate', 'BOD5 (mg/l)': 'BOD5',
    'CODcr(mg/l)': 'CODcr', 'SS(mg/l)': 'SS',
    'NH3-N(mg/l)': 'NH3-N', 'TP(mg/l)': 'TP',
    'TN(mg/l)': 'TN', 'Tin℃': 'Temp'
}
y_names_en = {
    'F/M(%)': 'F/M Ratio',
    'SVI': 'SVI',
    'SRT': 'SRT'
}

available_X = [col for col in X_columns if col in df.columns]
available_y = [col for col in y_columns if col in df.columns]

X_data = df[available_X].copy()
y_data = df[available_y].copy()

X_data = X_data.astype('float32')
y_data = y_data.astype('float32')

date_col = None
if '日期' in df.columns:
    date_col = '日期'
    df['日期'] = pd.to_datetime(df['日期'])

combined = pd.concat([X_data, y_data], axis=1).dropna()
X_data = combined[available_X]
y_data = combined[available_y]

if date_col:
    date_data = df.loc[combined.index, date_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# ============ 训练模型 ============
def train_models(X_data, y_data):
    X_scaled = scaler.fit_transform(X_data)
    models = {}
    results = {}
    
    for y_col in y_data.columns:
        y_target = y_data[y_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_target, test_size=0.2, random_state=42
        )
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        lasso = Lasso(alpha=0.1, random_state=42, max_iter=1000)
        lasso.fit(X_train, y_train)
        
        rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=20, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        models[y_col] = {
            'lr': lr, 'lasso': lasso, 'rf': rf, 'xgb': xgb_model,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
        
        results[y_col] = {}
        for name, model in [('lr', lr), ('lasso', lasso), ('rf', rf), ('xgb', xgb_model)]:
            y_pred = model.predict(X_test)
            results[y_col][name] = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
    
    return models, results

def predict_value(input_dict, model):
    input_array = np.array([input_dict[col] for col in available_X], dtype='float32').reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    return model.predict(input_scaled)[0]

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 📊 进水参数输入")
    st.markdown("---")
    
    input_values = {}
    for col in available_X:
        min_val = float(X_data[col].min())
        max_val = float(X_data[col].max())
        default_val = float(X_data[col].mean())
        input_values[col] = st.number_input(
            f"{x_names_cn.get(col, col)}",
            min_value=min_val, max_value=max_val,
            value=default_val,
            step=(max_val - min_val) / 100,
            format="%.2f"
        )
    
    st.markdown("---")
    
    if st.button("🚀 开始预测", use_container_width=True):
        st.session_state.predicted = True
        st.session_state.pred_values = {}
        st.session_state.input_values = input_values.copy()
        
        if not st.session_state.model_trained:
            with st.spinner("⏳ 训练模型中..."):
                models, results = train_models(X_data, y_data)
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.model_trained = True
        
        for y_col in available_y:
            model = st.session_state.models[y_col]['xgb']
            pred_val = predict_value(input_values, model)
            st.session_state.pred_values[y_col] = pred_val
        st.rerun()
    
    st.markdown("---")
    st.markdown("## 🎨 主题设置")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙 暗色", use_container_width=True):
            st.session_state.theme = 'dark'
            st.rerun()
    with col2:
        if st.button("☀️ 明亮", use_container_width=True):
            st.session_state.theme = 'light'
            st.rerun()
    
    current_theme = "🌙 暗色模式" if st.session_state.theme == 'dark' else "☀️ 明亮模式"
    st.markdown(f"<p style='text-align:center;color:{colors['text_secondary']};font-size:0.8rem;'>当前: {current_theme}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📁 导入数据")
    uploaded_file = st.file_uploader(
        "选择Excel文件",
        type=['xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_excel(uploaded_file, sheet_name=0)
            required_cols = ['日期'] + X_columns
            missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
            if missing_cols:
                st.warning(f"⚠️ 缺少列: {missing_cols[:3]}...")
            else:
                st.session_state.df_loaded = uploaded_df
                st.session_state.data_source = 'uploaded'
                st.session_state.model_trained = False
                st.success(f"✅ 成功导入 {len(uploaded_df)} 行数据！")
                st.info("🔄 请点击'开始预测'重新训练")
                if st.button("🔄 应用新数据"):
                    st.rerun()
        except Exception as e:
            st.error(f"❌ 读取失败: {str(e)}")
    
    if st.session_state.data_source == 'uploaded':
        st.info("📌 使用: 上传的数据")
    else:
        st.info("📌 使用: 默认数据")

# ============ 自定义正常范围 ============
FM_MIN, FM_MAX = 20.0, 40.0
SVI_MIN, SVI_MAX = 50.0, 150.0
SRT_MIN, SRT_MAX = 5.0, 15.0

# ============ 主区域 ============
if st.session_state.predicted and st.session_state.pred_values:
    pred_fm = st.session_state.pred_values.get('F/M(%)', 0)
    pred_svi = st.session_state.pred_values.get('SVI', 0)
    pred_srt = st.session_state.pred_values.get('SRT', 0)
    input_vals = st.session_state.input_values
    
    def get_status(val, min_val, max_val):
        if val < min_val:
            return "偏低", "status-warning"
        elif val > max_val:
            return "偏高", "status-danger"
        else:
            return "正常", "status-normal"
    
    fm_status, fm_class = get_status(pred_fm, FM_MIN, FM_MAX)
    svi_status, svi_class = get_status(pred_svi, SVI_MIN, SVI_MAX)
    srt_status, srt_class = get_status(pred_srt, SRT_MIN, SRT_MAX)
    
    raw_opt_srt = (pred_fm / 15.0) * 12.0
    opt_srt = max(SRT_MIN, min(SRT_MAX, raw_opt_srt))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">🧪 预测有机质占比</div>
            <div class="value">{pred_fm:.2f}%</div>
            <div class="sub"><span class="{fm_class}">{fm_status}</span></div>
            <div style="font-size:0.65rem;color:{colors['text_secondary']};">正常: {FM_MIN}% ~ {FM_MAX}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#f0883e;">
            <div class="label">📊 预测SVI</div>
            <div class="value">{pred_svi:.2f}</div>
            <div class="sub"><span class="{svi_class}">{svi_status}</span></div>
            <div style="font-size:0.65rem;color:{colors['text_secondary']};">正常: {SVI_MIN} ~ {SVI_MAX}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#3fb950;">
            <div class="label">⏳ 模型预测SRT</div>
            <div class="value">{pred_srt:.2f}<span style="font-size:0.9rem;color:{colors['text_secondary']};"> 天</span></div>
            <div class="sub"><span class="{srt_class}">{srt_status}</span></div>
            <div style="font-size:0.65rem;color:{colors['text_secondary']};">正常: {SRT_MIN} ~ {SRT_MAX} 天</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#d29922;">
            <div class="label">🌟 推荐最优污泥龄</div>
            <div class="value" style="color:#d29922;">{opt_srt:.2f}<span style="font-size:0.9rem;color:{colors['text_secondary']};"> 天</span></div>
            <div class="sub">基于F/M优化 (5~15天)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💾 导出预测结果")
    
    export_data = {'输入参数': [], '数值': []}
    for col, val in input_vals.items():
        export_data['输入参数'].append(x_names_cn.get(col, col))
        export_data['数值'].append(val)
    
    export_data['输入参数'].extend(['预测有机质占比(F/M)', '预测SVI', '预测SRT', '推荐最优污泥龄'])
    export_data['数值'].extend([f"{pred_fm:.2f}%", f"{pred_svi:.2f}", f"{pred_srt:.2f}天", f"{opt_srt:.2f}天"])
    export_data['输入参数'].extend(['有机质占比状态', 'SVI状态', 'SRT状态'])
    export_data['数值'].extend([fm_status, svi_status, srt_status])
    
    export_df = pd.DataFrame(export_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='预测结果', index=False)
        df_preview = df.head(20)
        df_preview.to_excel(writer, sheet_name='原始数据预览', index=False)
    
    st.download_button(
        label="📥 下载预测结果 (Excel)",
        data=output.getvalue(),
        file_name=f"预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
else:
    col1, col2, col3, col4 = st.columns(4)
    for col in [col1, col2, col3, col4]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="opacity:0.5;">
                <div class="label">等待预测...</div>
                <div class="value" style="font-size:1rem;color:{colors['text_secondary']};">点击"开始预测"</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ============ Tabs（5个并列） ============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 预测分析", "📈 时间序列", "📊 特征重要性",
    "📉 模型评价", "🔍 SHAP解释"
])

# ===== Tab 1: 预测分析 =====
with tab1:
    st.markdown("### 🎯 预测结果详情")
    if st.session_state.predicted and st.session_state.pred_values:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="result-card" style="border-top-color:#58a6ff;">
                <div class="label">🧪 有机质占比 (F/M)</div>
                <div class="value">{pred_fm:.2f}%</div>
                <div><span class="{fm_class}">{fm_status}</span></div>
                <div style="font-size:0.75rem;color:{colors['text_secondary']};margin-top:8px;">
                    正常范围: {FM_MIN}% ~ {FM_MAX}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card" style="border-top-color:#f0883e;">
                <div class="label">📊 SVI (污泥体积指数)</div>
                <div class="value">{pred_svi:.2f}</div>
                <div><span class="{svi_class}">{svi_status}</span></div>
                <div style="font-size:0.75rem;color:{colors['text_secondary']};margin-top:8px;">
                    正常范围: {SVI_MIN} ~ {SVI_MAX}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-card" style="border-top-color:#3fb950;">
                <div class="label">⏳ 污泥龄 (SRT)</div>
                <div class="value">{pred_srt:.2f} 天</div>
                <div><span class="{srt_class}">{srt_status}</span></div>
                <div style="font-size:0.75rem;color:{colors['text_secondary']};margin-top:8px;">
                    正常范围: {SRT_MIN} ~ {SRT_MAX} 天
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📍 SRT vs F/M 关系图")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_data['SRT'], y=y_data['F/M(%)'],
            mode='markers', name='历史数据',
            marker=dict(size=10, color='#58a6ff', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[pred_srt], y=[pred_fm],
            mode='markers', name='预测值',
            marker=dict(size=22, color='#f85149', symbol='star', line=dict(width=2, color='white' if st.session_state.theme == 'dark' else '#1a1a2e'))
        ))
        fig.update_layout(
            title='SRT vs F/M 关系图',
            xaxis_title='SRT (天)',
            yaxis_title='F/M (%)',
            height=350,
            hovermode='closest',
            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text_color'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 优化建议")
        
        if pred_fm > FM_MAX:
            st.warning(f"⚠️ 有机质占比偏高 ({pred_fm:.2f}%)，建议：减少进水量或增加MLSS浓度")
        elif pred_fm < FM_MIN:
            st.warning(f"⚠️ 有机质占比偏低 ({pred_fm:.2f}%)，建议：增加进水量或减少MLSS浓度")
        else:
            st.success(f"✅ 有机质占比正常 ({pred_fm:.2f}%)")
        
        if pred_srt > SRT_MAX:
            st.warning(f"⚠️ SRT偏高 ({pred_srt:.2f}天)，建议：减少污泥回流量，适当排泥")
        elif pred_srt < SRT_MIN:
            st.warning(f"⚠️ SRT偏低 ({pred_srt:.2f}天)，建议：增加污泥回流量")
        else:
            st.success(f"✅ SRT正常 ({pred_srt:.2f}天)")
        
        st.info(f"🌟 推荐最优污泥龄: **{opt_srt:.2f}天** (基于F/M={pred_fm:.1f}%优化)")
    else:
        st.info("💡 请先在左侧侧边栏输入参数，然后点击 '开始预测' 按钮")

# ===== Tab 2: 时间序列 =====
with tab2:
    st.markdown("### 📈 历史趋势分析")
    if date_col:
        time_target = st.selectbox(
            "选择指标查看时间序列",
            available_y + ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)'],
            format_func=lambda x: y_names_cn.get(x, x) if x in y_names_cn else x_names_cn.get(x, x),
            key="time_series"
        )
        if st.button("📊 生成时间序列图", key="gen_timeseries"):
            if time_target:
                if time_target in y_data.columns:
                    values = y_data[time_target]
                    title = y_names_cn.get(time_target, time_target)
                    color = '#58a6ff'
                else:
                    values = X_data[time_target]
                    title = x_names_cn.get(time_target, time_target)
                    color = '#f0883e'
                
                ma_window = st.slider("移动平均窗口", min_value=1, max_value=10, value=3, key="ma_window")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=date_data, y=values,
                    mode='lines+markers', name='原始数据',
                    line=dict(color=color, width=2), marker=dict(size=5, color=color)
                ))
                if ma_window > 1:
                    ma_values = values.rolling(window=ma_window).mean()
                    fig.add_trace(go.Scatter(
                        x=date_data, y=ma_values,
                        mode='lines', name=f'{ma_window}日移动平均',
                        line=dict(color='#f85149', width=3, dash='dash')
                    ))
                fig.update_layout(
                    title=f'{title} 时间序列趋势',
                    xaxis_title='日期',
                    yaxis_title=title,
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=colors['text_color'])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("当前值", f"{values.iloc[-1]:.2f}")
                with col2: st.metric("平均值", f"{values.mean():.2f}")
                with col3: st.metric("变化率", f"{((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100):.2f}%")
    else:
        st.warning("⚠️ 数据中未找到日期列")

# ===== Tab 3: 特征重要性 =====
with tab3:
    st.markdown("### 📊 特征重要性分析")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ 请先点击侧边栏的 '开始预测' 按钮训练模型")
    else:
        model_type = st.radio("选择模型", ['XGBoost', '随机森林', 'Lasso'], horizontal=True, key="importance")
        target = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key="importance_target")
        
        if st.button("📊 生成特征重要性图", key="gen_importance"):
            if target:
                model_key = {'XGBoost': 'xgb', '随机森林': 'rf', 'Lasso': 'lasso'}[model_type]
                if model_key == 'lasso':
                    importance = np.abs(st.session_state.models[target]['lasso'].coef_)
                else:
                    importance = st.session_state.models[target][model_key].feature_importances_
                
                sorted_idx = np.argsort(importance)[::-1]
                sorted_names = [x_names_en.get(available_X[i], available_X[i]) for i in sorted_idx]
                sorted_values = importance[sorted_idx]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                # 使用主题颜色
                bar_color = '#58a6ff' if st.session_state.theme == 'dark' else '#1a5276'
                text_color = colors['plot_textcolor']
                
                bars = ax.barh(sorted_names, sorted_values, color=bar_color)
                ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold', color=text_color)
                ax.set_title(f'{model_type} - {y_names_en.get(target, target)} Feature Importance', fontsize=13, fontweight='bold', color=text_color)
                ax.invert_yaxis()
                ax.set_facecolor(colors['plot_facecolor'])
                fig.patch.set_facecolor(colors['plot_facecolor'])
                for i, v in enumerate(sorted_values):
                    ax.text(v + 0.005, i, f'{v:.3f}', va='center', color=text_color, fontsize=9, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 🔥 特征相关性热力图")
        
        if st.button("📊 生成热力图", key="gen_heatmap"):
            corr_data = pd.concat([X_data, y_data], axis=1)
            corr_matrix = corr_data.corr()
            rename_map = {**x_names_en, **y_names_en}
            corr_matrix = corr_matrix.rename(columns=rename_map, index=rename_map)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, linewidths=0.5, ax=ax,
                       cbar_kws={'shrink': 0.8})
            text_color = colors['plot_textcolor']
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color=text_color)
            ax.set_facecolor(colors['plot_facecolor'])
            fig.patch.set_facecolor(colors['plot_facecolor'])
            plt.tight_layout()
            st.pyplot(fig)

# ===== Tab 4: 模型评价 =====
with tab4:
    st.markdown("### 📉 真实值 vs 预测值散点图")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ 请先点击侧边栏的 '开始预测' 按钮训练模型")
    else:
        target_eval = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key='eval')
        
        if st.button("📊 生成散点图", key="gen_scatter"):
            if target_eval:
                model = st.session_state.models[target_eval]['xgb']
                y_test = st.session_state.models[target_eval]['y_test']
                y_pred = model.predict(st.session_state.models[target_eval]['X_test'])
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                text_color = colors['plot_textcolor']
                ax.scatter(y_test, y_pred, alpha=0.6, color='#58a6ff', s=50)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
                ax.set_xlabel('True Value', fontsize=11, fontweight='bold', color=text_color)
                ax.set_ylabel('Predicted Value', fontsize=11, fontweight='bold', color=text_color)
                ax.set_title(f'{y_names_en.get(target_eval, target_eval)} - R² = {r2:.4f}', fontsize=13, fontweight='bold', color=text_color)
                ax.legend(loc='upper left', facecolor=colors['plot_facecolor'], edgecolor=colors['border'], labelcolor=text_color)
                ax.set_facecolor(colors['plot_facecolor'])
                fig.patch.set_facecolor(colors['plot_facecolor'])
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                st.markdown("### 📊 模型评价指标")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² 分数", f"{r2:.4f}", help="越接近1越好")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}", help="均方根误差，越小越好")
                with col3:
                    st.metric("MAE", f"{mae:.4f}", help="平均绝对误差，越小越好")
        
        st.markdown("---")
        st.markdown("### 📊 各模型性能对比")
        
        if st.button("📊 生成模型对比图", key="gen_compare"):
            if target_eval:
                model_names = ['Linear', 'Lasso', 'RF', 'XGB']
                r2_values = [st.session_state.results[target_eval]['lr']['r2'], 
                             st.session_state.results[target_eval]['lasso']['r2'],
                             st.session_state.results[target_eval]['rf']['r2'], 
                             st.session_state.results[target_eval]['xgb']['r2']]
                rmse_values = [st.session_state.results[target_eval]['lr']['rmse'], 
                               st.session_state.results[target_eval]['lasso']['rmse'],
                               st.session_state.results[target_eval]['rf']['rmse'], 
                               st.session_state.results[target_eval]['xgb']['rmse']]
                
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                text_color = colors['plot_textcolor']
                
                bars1 = ax1.bar(model_names, r2_values, color=['#58a6ff', '#f0883e', '#3fb950', '#f85149'])
                ax1.set_ylabel('R² Score', fontsize=11, color=text_color)
                ax1.set_title('R² Comparison', fontsize=13, fontweight='bold', color=text_color)
                ax1.set_ylim(0, 1.05)
                ax1.set_facecolor(colors['plot_facecolor'])
                fig2.patch.set_facecolor(colors['plot_facecolor'])
                for bar, val in zip(bars1, r2_values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', color=text_color, fontsize=9)
                
                bars2 = ax2.bar(model_names, rmse_values, color=['#58a6ff', '#f0883e', '#3fb950', '#f85149'])
                ax2.set_ylabel('RMSE', fontsize=11, color=text_color)
                ax2.set_title('RMSE Comparison', fontsize=13, fontweight='bold', color=text_color)
                ax2.set_facecolor(colors['plot_facecolor'])
                for bar, val in zip(bars2, rmse_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', color=text_color, fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig2)
        
        st.markdown("---")
        st.markdown("### 🎻 小提琴图 - 数据分布")
        
        target_violin = st.selectbox("选择变量查看分布", available_y + available_X[:4],
                                    format_func=lambda x: y_names_cn.get(x, x) if x in y_names_cn else x_names_cn.get(x, x),
                                    key='violin')
        
        if st.button("📊 生成小提琴图", key="gen_violin"):
            if target_violin:
                fig, ax = plt.subplots(figsize=(10, 4))
                if target_violin in y_data.columns:
                    data = y_data[target_violin]
                    title = y_names_en.get(target_violin, target_violin)
                else:
                    data = X_data[target_violin]
                    title = x_names_en.get(target_violin, target_violin)
                
                parts = ax.violinplot(data, positions=[1], showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor('#58a6ff')
                    pc.set_alpha(0.7)
                
                text_color = colors['plot_textcolor']
                ax.set_title(f'{title} Violin Plot', fontsize=13, fontweight='bold', color=text_color)
                ax.set_ylabel(title, fontsize=11, color=text_color)
                ax.set_xticks([1])
                ax.set_xticklabels([title], color=text_color)
                ax.grid(True, alpha=0.2)
                ax.set_facecolor(colors['plot_facecolor'])
                fig.patch.set_facecolor(colors['plot_facecolor'])
                plt.tight_layout()
                st.pyplot(fig)

# ===== Tab 5: SHAP解释 =====
with tab5:
    st.markdown("### 🔍 SHAP 模型解释")
    st.markdown("SHAP值解释每个特征对预测结果的贡献")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ 请先点击侧边栏的 '开始预测' 按钮训练模型")
    else:
        shap_target = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key='shap')
        
        if st.button("🎯 生成 SHAP 解释", key="shap_btn"):
            with st.spinner("⏳ 计算SHAP值中..."):
                try:
                    model = st.session_state.models[shap_target]['xgb']
                    X_train = st.session_state.models[shap_target]['X_train']
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train)
                    feature_names = [x_names_en.get(col, col) for col in available_X]
                    
                    st.markdown("#### 📊 SHAP 蜂群图")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.set_facecolor(colors['plot_facecolor'])
                    fig.patch.set_facecolor(colors['plot_facecolor'])
                    
                    shap.summary_plot(
                        shap_values, 
                        X_train, 
                        feature_names=feature_names,
                        show=False,
                        color_bar=True,
                        cmap=plt.get_cmap('coolwarm')
                    )
                    
                    ax = plt.gca()
                    text_color = colors['plot_textcolor']
                    ax.tick_params(colors=text_color, labelsize=10)
                    ax.xaxis.label.set_color(text_color)
                    ax.yaxis.label.set_color(text_color)
                    ax.title.set_color(text_color)
                    for text in ax.texts:
                        text.set_color(text_color)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("#### 📊 SHAP 特征重要性")
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.set_facecolor(colors['plot_facecolor'])
                    fig2.patch.set_facecolor(colors['plot_facecolor'])
                    
                    shap.summary_plot(
                        shap_values, 
                        X_train, 
                        feature_names=feature_names, 
                        plot_type="bar",
                        show=False,
                        color='#58a6ff' if st.session_state.theme == 'dark' else '#1a5276'
                    )
                    
                    ax2 = plt.gca()
                    ax2.tick_params(colors=text_color, labelsize=10)
                    ax2.xaxis.label.set_color(text_color)
                    ax2.yaxis.label.set_color(text_color)
                    ax2.title.set_color(text_color)
                    for patch in ax2.patches:
                        patch.set_color('#58a6ff' if st.session_state.theme == 'dark' else '#1a5276')
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    st.markdown("---")
                    st.markdown("#### 🎯 当前输入的SHAP解释")
                    
                    input_array = np.array([st.session_state.input_values.get(col, 0) for col in available_X], dtype='float32').reshape(1, -1)
                    input_scaled = scaler.transform(input_array)
                    
                    single_shap = explainer.shap_values(input_scaled)
                    
                    contrib_data = []
                    for i, name in enumerate(feature_names):
                        shap_val = single_shap[0][i]
                        contrib_data.append({
                            '特征': name,
                            'SHAP值': f"{shap_val:.3f}",
                            '影响方向': "⬆️ 正向" if shap_val > 0 else "⬇️ 负向"
                        })
                    
                    contrib_df = pd.DataFrame(contrib_data)
                    st.dataframe(contrib_df, use_container_width=True)
                    
                    pred_val = model.predict(input_scaled)[0]
                    base_val = explainer.expected_value
                    
                    st.markdown(f"""
                    <div style="background:{colors['card_bg']};padding:1rem;border-radius:10px;border:1px solid {colors['border']};margin-top:1rem;">
                        <b style="color:{colors['primary']};">预测 {y_names_cn.get(shap_target, shap_target)}:</b> 
                        <span style="color:{colors['text']};font-size:1.2rem;font-weight:bold;">{pred_val:.3f}</span>
                        <br>
                        <b style="color:{colors['text_secondary']};">基准值:</b> 
                        <span style="color:{colors['text']};">{base_val:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"SHAP 计算失败: {e}")

st.markdown("---")
st.markdown(f"💧 **污水处理智能分析平台 v6.0** | 轻量按需加载版 | {'🌙 暗色模式' if st.session_state.theme == 'dark' else '☀️ 明亮模式'}")
