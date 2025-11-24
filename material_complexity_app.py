import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from datetime import datetime
import gc

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
    """
    설정에 따라 엣지를 추출하는 통합 함수
    """
    # 1. 노이즈 제거 (가우시안 블러)
    # blur_k는 슬라이더 값 (0~5). 실제 커널 크기는 1, 3, 5, 7...
    if blur_k > 0:
        k_size = blur_k * 2 + 1
        blurred = cv2.GaussianBlur(image_gray, (k_size, k_size), 0)
    else:
        blurred = image_gray

    # 2. 엣지 검출 (모드별 분기)
    if mode == "Auto (Robust)":
        # Colab의 robust_canny 로직
        v = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blurred, lower, upper)
    else:
        # Manual 모드 (직접 설정)
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
    """엣지 맵으로 FD, L, R2 계산"""
    # 엣지 없으면 0 반환
    if np.sum(edges) < 100:
        return 1.0, 0.0, 0.0

    # 1. FD Calculation
    box_sizes = [2, 4, 8, 16, 32, 64]
    counts = []
    for size in box_sizes:
        counts.append(box_count(edges, int(size)))
    
    counts = np.array(counts)
    valid = counts > 0
    
    if np.sum(valid) < 2:
        return 1.0, 0.0, 0.0
        
    log_sizes = np.log(np.array(box_sizes)[valid])
    log_counts = np.log(np.array(counts)[valid])
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    slope = coeffs[0]
    
    # R-squared
    pred = slope * log_sizes + coeffs[1]
    ss_res = np.sum((log_counts - pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    FD = np.clip(-slope, 1.0, 2.0)

    # 2. Lacunarity Calculation (Edge based)
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
        # 0~1 정규화 (Colab과 동일 로직)
        L_norm = 1 - (1 / (1 + L_val))

    return FD, L_norm, r2

# -----------------------------------------------------------------------------
# 2. UI 구성
# -----------------------------------------------------------------------------

st.title("🔬 Material Complexity Analyzer")
st.markdown("이미지 특성에 맞춰 **자동(Auto)** 또는 **수동(Manual)** 모드를 선택하세요.")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    
    mode = st.radio("분석 모드", ["Auto (Robust)", "Manual (Tuning)"],
                    help="Auto: 대부분의 사진에 적합\nManual: 엣지가 안 잡히거나 너무 지글거릴 때 사용")
    
    st.divider()
    
    blur_val = 0
    sigma_val = 0.33
    canny_th = (50, 150)
    
    if mode == "Auto (Robust)":
        st.info("💡 **자동 모드**\n노이즈를 제거하고 밝기에 따라 엣지를 자동으로 검출합니다.")
        # 자동 모드에서도 미세 조정 가능하게 함
        sigma_val = st.slider("민감도 (Sigma)", 0.1, 1.0, 0.33, 
                              help="낮으면 엄격하게, 높으면 헐렁하게 잡습니다.")
        blur_val = 1 # 기본 블러 켜기
        
    else:
        st.info("💡 **수동 모드 (정밀 튜닝)**\n눈으로 확인하며 직접 조절하세요.")
        blur_val = st.slider("가우시안 블러 (노이즈 제거)", 0, 5, 1)
        canny_th = st.slider("Canny 임계값 (Min, Max)", 0, 255, (30, 150))

    st.divider()
    if st.button("🗑️ 기록 초기화"):
        st.session_state.history = []
        st.rerun()
    
    # 히스토리 표시
    if st.session_state.history:
        st.subheader("최근 기록")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df[['name', 'FD', 'L']], use_container_width=True)


# --- 메인 화면 ---

# 파일 업로드 (다중 허용)
uploaded_files = st.file_uploader("이미지 업로드", type=['jpg', 'png'], accept_multiple_files=True)

if uploaded_files:
    # ---------------------------------------------------------
    # A. 미리보기 및 튜닝 (첫 번째 이미지 기준)
    # ---------------------------------------------------------
    st.subheader("1️⃣ 엣지 검출 미리보기")
    
    # 튜닝을 위해 첫 번째 이미지만 먼저 로드
    first_file = uploaded_files[0]
    img_pil = Image.open(first_file)
    img_pil = resize_for_memory(img_pil)
    img_np = np.array(img_pil)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 현재 설정으로 엣지 추출
    edges_preview = get_edges(img_gray, mode, blur_val, sigma_val, canny_th[0], canny_th[1])
    
    # 미리보기 컬럼
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_pil, caption=f"원본: {first_file.name}", use_container_width=True)
    with c2:
        density = (np.sum(edges_preview>0)/edges_preview.size)*100
        st.image(edges_preview, caption=f"검출된 엣지 (Density: {density:.1f}%)", use_container_width=True)
        
        # 간단 가이드 메시지
        if density < 1:
            st.warning("⚠️ 엣지가 거의 없습니다! 민감도(Sigma)를 높이거나 임계값을 낮추세요.")
        elif density > 25:
            st.warning("⚠️ 너무 복잡합니다(지글거림). 블러를 높이세요.")
        else:
            st.success("✅ 적절한 검출 상태입니다.")

    # ---------------------------------------------------------
    # B. 분석 실행
    # ---------------------------------------------------------
    st.divider()
    st.subheader("2️⃣ 분석 실행")
    
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        run_btn = st.button("🚀 전체 분석 시작", type="primary", use_container_width=True)
    
    if run_btn:
        results_container = st.container()
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            try:
                # 이미지 로드 & 리사이징
                c_img = Image.open(file)
                c_img = resize_for_memory(c_img)
                c_img_np = np.array(c_img)
                c_gray = cv2.cvtColor(c_img_np, cv2.COLOR_RGB2GRAY)
                
                # 설정된 파라미터로 엣지 추출
                edges = get_edges(c_gray, mode, blur_val, sigma_val, canny_th[0], canny_th[1])
                
                # 지표 계산
                FD, L, r2 = calc_metrics(edges)
                
                # 결과 저장
                st.session_state.history.append({
                    'name': file.name,
                    'FD': round(FD, 4),
                    'L': round(L, 4),
                    'R2': round(r2, 4),
                    'Density': f"{(np.sum(edges>0)/edges.size)*100:.1f}%"
                })
                
                # 메모리 정리
                del c_img, c_img_np, c_gray, edges
                gc.collect()
                
            except Exception as e:
                st.error(f"{file.name} 분석 중 오류: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.success("모든 분석이 완료되었습니다!")
        
        # 최종 결과 테이블
        if st.session_state.history:
            df_res = pd.DataFrame(st.session_state.history)
            st.dataframe(df_res.iloc[::-1], use_container_width=True)
            
            # CSV 다운로드
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 결과 CSV 다운로드",
                csv,
                "complexity_results.csv",
                "text/csv"
            )
