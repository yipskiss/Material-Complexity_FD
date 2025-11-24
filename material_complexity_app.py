import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from datetime import datetime
import gc
import os
import matplotlib.pyplot as plt # 그래프 생성을 위해 추가

# 대용량 이미지 경고 무시
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(
    page_title="재질 복잡도 측정기 (Ultimate)",
    page_icon="🔬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. 핵심 알고리즘
# -----------------------------------------------------------------------------

def resize_for_memory(image, max_dim=1024):
    width, height = image.size
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def get_edges(image_gray, mode="Auto", blur_k=1, sigma=0.33, low_th=50, high_th=150):
    """설정에 따라 엣지를 추출하는 통합 함수"""
    if blur_k > 0:
        k_size = blur_k * 2 + 1
        blurred = cv2.GaussianBlur(image_gray, (k_size, k_size), 0)
    else:
        blurred = image_gray

    if mode == "Auto (Robust)":
        v = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blurred, lower, upper)
    else:
        edged = cv2.Canny(blurred, low_th, high_th)
        
    return edged

def box_count(edges, k):
    S = edges.shape
    h_trim = S[0] // k * k
    w_trim = S[1] // k * k
    if h_trim == 0 or w_trim == 0: return 0
    img_trim = edges[:h_trim, :w_trim]
    reshaped = img_trim.reshape(h_trim//k, k, w_trim//k, k)
    has_edge = np.max(reshaped, axis=(1, 3)) > 0
    return np.sum(has_edge)

def calc_metrics(edges):
    """
    FD, L, R2 및 그래프용 플롯 데이터를 반환하도록 수정됨
    """
    # 엣지가 너무 없으면 기본값 반환
    if np.sum(edges) < 100:
        return 1.0, 0.0, 0.0, None

    box_sizes = [2, 4, 8, 16, 32, 64]
    counts = []
    for size in box_sizes:
        counts.append(box_count(edges, int(size)))
    
    counts = np.array(counts)
    valid = counts > 0
    
    if np.sum(valid) < 2:
        return 1.0, 0.0, 0.0, None
        
    log_sizes = np.log(np.array(box_sizes)[valid])
    log_counts = np.log(np.array(counts)[valid])
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    slope = coeffs[0]
    
    # R-squared 계산
    pred = slope * log_sizes + coeffs[1]
    ss_res = np.sum((log_counts - pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    FD = np.clip(-slope, 1.0, 2.0)

    # Lacunarity Calculation
    box_size, stride = 32, 16
    h, w = edges.shape
    masses = []
    for i in range(0, h-box_size, stride):
        for j in range(0, w-box_size, stride):
            masses.append(np.sum(edges[i:i+box_size, j:j+box_size] > 0))
            
    masses = np.array(masses)
    if len(masses) == 0 or np.mean(masses) == 0:
        L_norm = 0.0
    else:
        L_val = (np.std(masses) / np.mean(masses)) ** 2
        L_norm = 1 - (1 / (1 + L_val))

    # 그래프를 그리기 위한 데이터 딕셔너리 생성
    plot_data = {
        'x': log_sizes,
        'y': log_counts,
        'pred': pred,
        'equation': f"y = {slope:.3f}x + {coeffs[1]:.3f}"
    }

    return FD, L_norm, r2, plot_data

# -----------------------------------------------------------------------------
# 2. UI 구성
# -----------------------------------------------------------------------------

st.title("🔬 Material Complexity Analyzer")
st.markdown("이미지 특성에 맞춰 **자동(Auto)** 또는 **수동(Manual)** 모드를 선택하세요.")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- 탭 구성 (분석기 / 설명서) ---
tab_analyzer, tab_readme = st.tabs(["📊 분석기 (Analyzer)", "📖 설명서 (Manual)"])


# =========================================================
# TAB 1: 분석기 (기존 기능)
# =========================================================
with tab_analyzer:
    
    # --- 사이드바 ---
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        
        mode = st.radio("분석 모드", ["Auto (Robust)", "Manual (Tuning)"],
                        help="Auto: Sigma(비율)로 자동 계산\nManual: 임계값 숫자를 직접 지정")
        
        st.divider()
        
        blur_val = 1
        sigma_val = 0.33
        canny_th = (50, 150)
        
        if mode == "Auto (Robust)":
            st.info("💡 **자동 모드**")
            blur_val = st.slider("가우시안 블러", 0, 5, 1)
            sigma_val = st.slider("민감도 (Sigma)", 0.1, 1.0, 0.33)
        else:
            st.info("💡 **수동 모드**")
            blur_val = st.slider("가우시안 블러", 0, 5, 1)
            canny_th = st.slider("Canny 임계값", 0, 255, (30, 150))

        st.divider()
        if st.button("🗑️ 기록 초기화"):
            st.session_state.history = []
            st.rerun()
        
        # [수정] R2도 사이드바 표에 포함시킴
        if st.session_state.history:
            st.subheader("최근 기록")
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df[['name', 'FD', 'L', 'R2']], use_container_width=True)

    # --- 메인 화면 ---
    uploaded_files = st.file_uploader("이미지 업로드", type=['jpg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        st.subheader("1️⃣ 엣지 검출 미리보기")
        
        first_file = uploaded_files[0]
        img_pil = Image.open(first_file)
        img_pil = resize_for_memory(img_pil)
        img_np = np.array(img_pil)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        edges_preview = get_edges(img_gray, mode, blur_val, sigma_val, canny_th[0], canny_th[1])
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_pil, caption=f"원본: {first_file.name}", use_container_width=True)
        with c2:
            density = (np.sum(edges_preview>0)/edges_preview.size)*100
            st.image(edges_preview, caption=f"검출된 엣지 (Density: {density:.1f}%)", use_container_width=True)
            
            if density < 1:
                st.warning("⚠️ 엣지가 없습니다! 설정을 조절하세요.")
            elif density > 25:
                st.warning("⚠️ 너무 지글거립니다! 블러를 높이세요.")
            else:
                st.success("✅ 적절한 검출 상태입니다.")

        st.divider()
        st.subheader("2️⃣ 분석 실행")
        
        if st.button("🚀 전체 분석 시작", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                try:
                    c_img = Image.open(file)
                    c_img = resize_for_memory(c_img)
                    c_img_np = np.array(c_img)
                    c_gray = cv2.cvtColor(c_img_np, cv2.COLOR_RGB2GRAY)
                    
                    edges = get_edges(c_gray, mode, blur_val, sigma_val, canny_th[0], canny_th[1])
                    
                    # [수정] plot_data도 함께 받아옴
                    FD, L, r2, plot_data = calc_metrics(edges)
                    
                    # 결과 저장
                    st.session_state.history.append({
                        'name': file.name,
                        'FD': round(FD, 4),
                        'L': round(L, 4),
                        'R2': round(r2, 4),
                        'Density': f"{(np.sum(edges>0)/edges.size)*100:.1f}%"
                    })
                    
                    # --- [추가] 개별 분석 결과 바로 보여주기 (그래프 포함) ---
                    with st.expander(f"📊 {file.name} - 상세 결과 보기 (R²: {r2:.4f})", expanded=True):
                        col_metrics, col_plot = st.columns([1, 2])
                        
                        with col_metrics:
                            st.metric("Fractal Dimension (FD)", f"{FD:.4f}")
                            st.metric("Lacunarity (L)", f"{L:.4f}")
                            st.metric("R-Squared (신뢰도)", f"{r2:.4f}")
                            if r2 < 0.9:
                                st.error("⚠️ 신뢰도 낮음: 데이터가 직선에 맞지 않습니다.")
                            else:
                                st.success("✅ 신뢰도 높음")

                        with col_plot:
                            if plot_data:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.scatter(plot_data['x'], plot_data['y'], color='blue', alpha=0.6, label='Data Points')
                                ax.plot(plot_data['x'], plot_data['pred'], color='red', linestyle='--', label='Linear Fit')
                                ax.set_xlabel("Log(Box Size)")
                                ax.set_ylabel("Log(Count)")
                                ax.set_title(f"Log-Log Plot (Slope: -{FD:.3f})")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig) # 메모리 해제
                            else:
                                st.warning("데이터가 부족하여 그래프를 그릴 수 없습니다.")

                    del c_img, c_img_np, c_gray, edges
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"{file.name} 분석 중 오류: {e}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.success("모든 분석이 완료되었습니다!")
            
            # 하단 통합 테이블
            if st.session_state.history:
                st.divider()
                st.subheader("📋 전체 결과 요약")
                df_res = pd.DataFrame(st.session_state.history)
                st.dataframe(df_res.iloc[::-1], use_container_width=True)
                
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 결과 CSV 다운로드", csv, "complexity_results.csv", "text/csv")


# =========================================================
# TAB 2: 설명서 (README.md 표시)
# =========================================================
with tab_readme:
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    else:
        st.warning("⚠️ README.md 파일을 찾을 수 없습니다.")
        st.info("같은 폴더에 README.md 파일을 업로드해주세요.")
