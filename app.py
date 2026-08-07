# ===== Fix for Streamlit Cloud =====
import matplotlib
matplotlib.use('Agg')
# ==================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import shap

# ============ 页面配置 ============
st.set_page_config(page_title="污水处理智能分析平台", page_icon="💧", layout="wide")

# ============ 初始化状态 ============
for key in ['models', 'results', 'model_trained', 'predicted', 'pred_values', 'input_values', 'theme']:
    if key not in st.session_state:
        if key == 'theme':
            st.session_state[key] = 'dark'
        elif key in ['model_trained', 'predicted']:
            st.session_state[key] = False
        else:
            st.session_state[key] = None if key in ['models', 'results', 'input_values'] else {}

# 图表状态
chart_keys = ['show_ts', 'show_imp', 'show_heat', 'show_scatter', 'show_compare', 'show_violin', 'show_shap']
for key in chart_keys:
    if key not in st.session_state:
        st.session_state[key] = False

# ============ 主题配色 ============
def get_colors(theme):
    if theme == 'dark':
        return {'face': '#0d1117', 'text': 'white', 'bar': '#58a6ff', 'bg': '#0e1117'}
    else:
        return {'face': '#ffffff', 'text': '#1a1a2e', 'bar': '#1a5276', 'bg': '#f5f7fa'}

colors = get_colors(st.session_state.theme)

# 设置matplotlib
plt.rcParams['text.color'] = colors['text']
plt.rcParams['axes.labelcolor'] = colors['text']
plt.rcParams['xtick.color'] = colors['text']
plt.rcParams['ytick.color'] = colors['text']
plt.rcParams['figure.facecolor'] = colors['face']
plt.rcParams['axes.facecolor'] = colors['face']

# ============ CSS ============
st.markdown(f"""
<style>
.stApp {{ background-color: {colors['bg']}; }}
section[data-testid="stSidebar"] {{ background-color: #0d1117; border-right: 1px solid #30363d; }}
.main-header {{ font-size: 2.5rem; font-weight: 700; color: #58a6ff; text-align: center; }}
.sub-header {{ font-size: 1rem; color: #8b949e; text-align: center; border-bottom: 1px solid #30363d; padding-bottom: 1rem; }}
.metric-card {{ background: #161b22; border-radius: 10px; padding: 1rem; border-left: 4px solid #58a6ff; text-align: center; }}
.metric-card .value {{ font-size: 1.8rem; font-weight: 700; color: #f0f6fc; }}
.metric-card .label {{ font-size: 0.75rem; color: #8b949e; }}
.status-normal {{ color: #3fb950; font-weight: 700; }}
.status-warning {{ color: #d29922; font-weight: 700; }}
.status-danger {{ color: #f85149; font-weight: 700; }}
.stButton button {{ background: #238636; color: white; font-weight: 700; border: none; border-radius: 8px; padding: 0.6rem 2rem; width: 100%; }}
.stButton button:hover {{ background: #2ea043; }}
.chart-container {{ background: {colors['face']}; border-radius: 10px; padding: 1rem; margin-top: 1rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💧 污水处理智能分析平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于进水参数的污泥指标预测与SRT优化系统</div>', unsafe_allow_html=True)

# ============ 加载数据 ============
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('随机森林归一化.xlsx', sheet_name='Sheet1')
        return df
    except:
        return None

df = load_data()
if df is None:
    st.error("❌ 找不到数据文件！")
    st.stop()

X_cols = ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)', 'SS(mg/l)', 'NH3-N(mg/l)', 'TP(mg/l)', 'TN(mg/l)', 'Tin℃']
y_cols = ['F/M(%)', 'SVI', 'SRT']

x_cn = {'Qoutm3/d': '进水流量', 'BOD5 (mg/l)': '进水BOD5', 'CODcr(mg/l)': '进水CODcr', 'SS(mg/l)': '进水SS', 'NH3-N(mg/l)': '进水NH3-N', 'TP(mg/l)': '进水TP', 'TN(mg/l)': '进水TN', 'Tin℃': '进水水温'}
y_cn = {'F/M(%)': '有机质占比', 'SVI': 'SVI (污泥体积指数)', 'SRT': 'SRT (污泥龄)'}
x_en = {'Qoutm3/d': 'Flow Rate', 'BOD5 (mg/l)': 'BOD5', 'CODcr(mg/l)': 'CODcr', 'SS(mg/l)': 'SS', 'NH3-N(mg/l)': 'NH3-N', 'TP(mg/l)': 'TP', 'TN(mg/l)': 'TN', 'Tin℃': 'Temp'}
y_en = {'F/M(%)': 'F/M Ratio', 'SVI': 'SVI', 'SRT': 'SRT'}

X_data = df[X_cols].astype('float32')
y_data = df[y_cols].astype('float32')
date_col = df['日期'] if '日期' in df.columns else None

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# ============ 训练模型 ============
def train_models():
    models, results = {}, {}
    for yc in y_cols:
        yt = y_data[yc].values
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, yt, test_size=0.2, random_state=42)
        lr = LinearRegression().fit(X_train, y_train)
        la = Lasso(alpha=0.1, random_state=42, max_iter=1000).fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1).fit(X_train, y_train)
        xg = xgb.XGBRegressor(n_estimators=20, max_depth=4, random_state=42, verbosity=0).fit(X_train, y_train)
        models[yc] = {'lr': lr, 'lasso': la, 'rf': rf, 'xgb': xg, 'X_train': X_train, 'X_test': X_test, 'y_test': y_test}
        results[yc] = {}
        for name, m in [('lr', lr), ('lasso', la), ('rf', rf), ('xgb', xg)]:
            yp = m.predict(X_test)
            results[yc][name] = {'r2': r2_score(y_test, yp), 'mse': mean_squared_error(y_test, yp), 'rmse': np.sqrt(mean_squared_error(y_test, yp)), 'mae': mean_absolute_error(y_test, yp)}
    return models, results

def predict_val(input_dict, model):
    arr = np.array([input_dict[c] for c in X_cols], dtype='float32').reshape(1, -1)
    return model.predict(scaler.transform(arr))[0]

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 📊 进水参数输入")
    inputs = {}
    for c in X_cols:
        inputs[c] = st.number_input(x_cn[c], value=float(X_data[c].mean()), format="%.2f")
    
    if st.button("🚀 开始预测", use_container_width=True):
        if not st.session_state.model_trained:
            with st.spinner("训练中..."):
                st.session_state.models, st.session_state.results = train_models()
                st.session_state.model_trained = True
        st.session_state.predicted = True
        st.session_state.pred_values = {yc: predict_val(inputs, st.session_state.models[yc]['xgb']) for yc in y_cols}
        st.session_state.input_values = inputs
        st.rerun()
    
    st.markdown("---")
    st.markdown("## 🎨 主题")
    c1, c2 = st.columns(2)
    if c1.button("🌙 暗色"): st.session_state.theme = 'dark'; st.rerun()
    if c2.button("☀️ 明亮"): st.session_state.theme = 'light'; st.rerun()

# ============ 显示预测结果 ============
FM_MIN, FM_MAX, SVI_MIN, SVI_MAX, SRT_MIN, SRT_MAX = 20, 40, 50, 150, 5, 15

if st.session_state.predicted and st.session_state.pred_values:
    fm, svi, srt = st.session_state.pred_values['F/M(%)'], st.session_state.pred_values['SVI'], st.session_state.pred_values['SRT']
    def stat(v, mn, mx): return "正常" if mn <= v <= mx else ("偏高" if v > mx else "偏低")
    def stat_class(v, mn, mx): return "status-normal" if mn <= v <= mx else ("status-danger" if v > mx else "status-warning")
    
    opt_srt = max(SRT_MIN, min(SRT_MAX, (fm / 15) * 12))
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="label">🧪 有机质占比</div><div class="value">{fm:.2f}%</div><div><span class="{stat_class(fm, FM_MIN, FM_MAX)}">{stat(fm, FM_MIN, FM_MAX)}</span></div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card" style="border-left-color:#f0883e;"><div class="label">📊 SVI</div><div class="value">{svi:.2f}</div><div><span class="{stat_class(svi, SVI_MIN, SVI_MAX)}">{stat(svi, SVI_MIN, SVI_MAX)}</span></div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card" style="border-left-color:#3fb950;"><div class="label">⏳ SRT</div><div class="value">{srt:.2f}天</div><div><span class="{stat_class(srt, SRT_MIN, SRT_MAX)}">{stat(srt, SRT_MIN, SRT_MAX)}</span></div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card" style="border-left-color:#d29922;"><div class="label">🌟 推荐污泥龄</div><div class="value" style="color:#d29922;">{opt_srt:.2f}天</div></div>', unsafe_allow_html=True)

# ============ Tabs ============
t1, t2, t3, t4, t5 = st.tabs(["📊 预测分析", "📈 时间序列", "📊 特征重要性", "📉 模型评价", "🔍 SHAP解释"])

# ===== Tab 1: 预测分析 =====
with t1:
    if st.session_state.predicted:
        st.markdown(f"**有机质占比**: {fm:.2f}% ({stat(fm, FM_MIN, FM_MAX)}) | **SVI**: {svi:.2f} ({stat(svi, SVI_MIN, SVI_MAX)}) | **SRT**: {srt:.2f}天 ({stat(srt, SRT_MIN, SRT_MAX)})")
        if fm > FM_MAX: st.warning(f"⚠️ 有机质占比偏高 ({fm:.2f}%)，建议减少进水量或增加MLSS")
        elif fm < FM_MIN: st.warning(f"⚠️ 有机质占比偏低 ({fm:.2f}%)，建议增加进水量或减少MLSS")
        else: st.success(f"✅ 有机质占比正常 ({fm:.2f}%)")
        if srt > SRT_MAX: st.warning(f"⚠️ SRT偏高 ({srt:.2f}天)，建议减少污泥回流量")
        elif srt < SRT_MIN: st.warning(f"⚠️ SRT偏低 ({srt:.2f}天)，建议增加污泥回流量")
        else: st.success(f"✅ SRT正常 ({srt:.2f}天)")
        st.info(f"🌟 推荐最优污泥龄: **{opt_srt:.2f}天**")
        
        # SRT vs F/M 关系图
        st.markdown("---")
        st.markdown("### 📍 SRT vs F/M 关系图")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_data['SRT'], y=y_data['F/M(%)'],
            mode='markers', name='历史数据',
            marker=dict(size=10, color='#58a6ff', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[srt], y=[fm],
            mode='markers', name='预测值',
            marker=dict(size=22, color='#f85149', symbol='star', line=dict(width=2, color='white' if st.session_state.theme == 'dark' else '#1a1a2e'))
        ))
        fig.update_layout(
            title='SRT vs F/M 关系图',
            xaxis_title='SRT (天)',
            yaxis_title='F/M (%)',
            height=350,
            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text'])
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 请先点击侧边栏 '开始预测'")

# ===== Tab 2: 时间序列 =====
with t2:
    if date_col is not None:
        target = st.selectbox("选择指标", y_cols + ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)'], format_func=lambda x: y_cn.get(x, x) if x in y_cn else x_cn.get(x, x))
        c1, c2 = st.columns([3, 1])
        if c1.button("📊 生成时间序列图", key="gen_ts"): st.session_state.show_ts = True; st.rerun()
        if c2.button("🗑️", key="clr_ts"): st.session_state.show_ts = False; st.rerun()
        if st.session_state.show_ts:
            vals = y_data[target] if target in y_cols else X_data[target]
            title = y_cn.get(target, x_cn.get(target, target))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=date_col, y=vals, mode='lines+markers', name='原始数据'))
            fig.update_layout(title=f'{title} 趋势', xaxis_title='日期', yaxis_title=title, height=350, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 无日期列")

# ===== Tab 3: 特征重要性 =====
with t3:
    if st.session_state.model_trained:
        mt = st.radio("模型", ['XGBoost', '随机森林', 'Lasso'], horizontal=True)
        tg = st.selectbox("目标", y_cols, format_func=lambda x: y_cn[x])
        c1, c2 = st.columns([3, 1])
        if c1.button("📊 生成特征重要性", key="gen_imp"): st.session_state.show_imp = True; st.rerun()
        if c2.button("🗑️", key="clr_imp"): st.session_state.show_imp = False; st.rerun()
        if st.session_state.show_imp:
            key = {'XGBoost':'xgb','随机森林':'rf','Lasso':'lasso'}[mt]
            imp = np.abs(st.session_state.models[tg]['lasso'].coef_) if key == 'lasso' else st.session_state.models[tg][key].feature_importances_
            idx = np.argsort(imp)[::-1]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh([x_en[X_cols[i]] for i in idx], imp[idx], color=colors['bar'])
            ax.set_xlabel('Importance', color=colors['text'])
            ax.set_title(f'{mt} - {tg} Feature Importance', color=colors['text'])
            ax.set_facecolor(colors['face'])
            fig.patch.set_facecolor(colors['face'])
            st.pyplot(fig)
        
        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        if c1.button("📊 生成热力图", key="gen_heat"): st.session_state.show_heat = True; st.rerun()
        if c2.button("🗑️", key="clr_heat"): st.session_state.show_heat = False; st.rerun()
        if st.session_state.show_heat:
            corr = pd.concat([X_data, y_data], axis=1).corr()
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax)
            ax.set_title('Correlation Heatmap', color=colors['text'])
            ax.set_facecolor(colors['face'])
            fig.patch.set_facecolor(colors['face'])
            st.pyplot(fig)
    else:
        st.warning("⚠️ 请先训练模型")

# ===== Tab 4: 模型评价 =====
with t4:
    if st.session_state.model_trained:
        tg = st.selectbox("目标", y_cols, format_func=lambda x: y_cn[x], key='eval')
        
        # 散点图
        c1, c2 = st.columns([3, 1])
        if c1.button("📊 生成散点图", key="gen_scatter"): st.session_state.show_scatter = True; st.rerun()
        if c2.button("🗑️", key="clr_scatter"): st.session_state.show_scatter = False; st.rerun()
        if st.session_state.show_scatter:
            m = st.session_state.models[tg]['xgb']
            yt = st.session_state.models[tg]['y_test']
            yp = m.predict(st.session_state.models[tg]['X_test'])
            r2 = r2_score(yt, yp)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(yt, yp, alpha=0.6, color='#58a6ff', s=50)
            # 绘制理想线
            min_val = min(yt.min(), yp.min())
            max_val = max(yt.max(), yp.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
            
            # 使用不同的标题和标签
            y_name_cn = y_cn.get(tg, tg)
            y_name_en = y_en.get(tg, tg)
            ax.set_xlabel(f'True {y_name_en}', color=colors['text'], fontsize=11)
            ax.set_ylabel(f'Predicted {y_name_en}', color=colors['text'], fontsize=11)
            ax.set_title(f'{y_name_cn} - R² = {r2:.4f}', color=colors['text'], fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', facecolor=colors['face'], edgecolor='#30363d', labelcolor=colors['text'])
            ax.set_facecolor(colors['face'])
            fig.patch.set_facecolor(colors['face'])
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"**R²**: {r2:.4f} | **MSE**: {mean_squared_error(yt, yp):.4f} | **RMSE**: {np.sqrt(mean_squared_error(yt, yp)):.4f} | **MAE**: {mean_absolute_error(yt, yp):.4f}")
        
        # 模型对比
        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        if c1.button("📊 生成模型对比", key="gen_compare"): st.session_state.show_compare = True; st.rerun()
        if c2.button("🗑️", key="clr_compare"): st.session_state.show_compare = False; st.rerun()
        if st.session_state.show_compare:
            names, r2s, rmses = ['Linear','Lasso','RF','XGB'], [], []
            for n in ['lr','lasso','rf','xgb']:
                r2s.append(st.session_state.results[tg][n]['r2'])
                rmses.append(st.session_state.results[tg][n]['rmse'])
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
            a1.bar(names, r2s, color=['#58a6ff','#f0883e','#3fb950','#f85149'])
            a1.set_ylabel('R²', color=colors['text'])
            a1.set_title(f'{tg} - R² Comparison', color=colors['text'])
            for i, v in enumerate(r2s):
                a1.text(i, v + 0.01, f'{v:.3f}', ha='center', color=colors['text'])
            a2.bar(names, rmses, color=['#58a6ff','#f0883e','#3fb950','#f85149'])
            a2.set_ylabel('RMSE', color=colors['text'])
            a2.set_title(f'{tg} - RMSE Comparison', color=colors['text'])
            for i, v in enumerate(rmses):
                a2.text(i, v + 0.01, f'{v:.3f}', ha='center', color=colors['text'])
            a1.set_facecolor(colors['face'])
            a2.set_facecolor(colors['face'])
            fig.patch.set_facecolor(colors['face'])
            st.pyplot(fig)
        
        # 小提琴图
        st.markdown("---")
        tv = st.selectbox("选择变量查看分布", y_cols + ['CODcr(mg/l)', 'SS(mg/l)'], format_func=lambda x: y_cn.get(x, x) if x in y_cn else x_cn.get(x, x), key='violin')
        c1, c2 = st.columns([3, 1])
        if c1.button("📊 生成小提琴图", key="gen_violin"): st.session_state.show_violin = True; st.rerun()
        if c2.button("🗑️", key="clr_violin"): st.session_state.show_violin = False; st.rerun()
        if st.session_state.show_violin:
            vals = y_data[tv] if tv in y_cols else X_data[tv]
            title = y_cn.get(tv, x_cn.get(tv, tv))
            fig, ax = plt.subplots(figsize=(10, 4))
            parts = ax.violinplot(vals, positions=[1], showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('#58a6ff')
                pc.set_alpha(0.7)
            ax.set_title(f'{title} 分布小提琴图', color=colors['text'])
            ax.set_xticks([1])
            ax.set_xticklabels([title], color=colors['text'])
            ax.set_ylabel('数值', color=colors['text'])
            ax.grid(True, alpha=0.2)
            ax.set_facecolor(colors['face'])
            fig.patch.set_facecolor(colors['face'])
            st.pyplot(fig)
    else:
        st.warning("⚠️ 请先训练模型")

# ===== Tab 5: SHAP解释 =====
with t5:
    if st.session_state.model_trained:
        tg = st.selectbox("目标", y_cols, format_func=lambda x: y_cn[x], key='shap')
        c1, c2 = st.columns([3, 1])
        if c1.button("🎯 生成SHAP解释", key="gen_shap"): st.session_state.show_shap = True; st.rerun()
        if c2.button("🗑️", key="clr_shap"): st.session_state.show_shap = False; st.rerun()
        if st.session_state.show_shap:
            with st.spinner("计算中..."):
                m = st.session_state.models[tg]['xgb']
                Xt = st.session_state.models[tg]['X_train']
                e = shap.TreeExplainer(m)
                sv = e.shap_values(Xt)
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.summary_plot(sv, Xt, feature_names=[x_en[c] for c in X_cols], show=False)
                ax.set_facecolor(colors['face'])
                fig.patch.set_facecolor(colors['face'])
                st.pyplot(fig)
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                shap.summary_plot(sv, Xt, feature_names=[x_en[c] for c in X_cols], plot_type="bar", show=False)
                ax2.set_facecolor(colors['face'])
                fig2.patch.set_facecolor(colors['face'])
                st.pyplot(fig2)
    else:
        st.warning("⚠️ 请先训练模型")
