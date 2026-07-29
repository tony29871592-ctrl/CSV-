import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==========================================
# 0. 페이지 설정 및 세션 상태 초기화
# ==========================================
st.set_page_config(
    page_title="CSV 데이터로 배우는 선형회귀 실험실",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 상태(Session State) 초기화
if "df" not in st.session_state:
    st.session_state["df"] = None
if "simple_results" not in st.session_state:
    st.session_state["simple_results"] = None
if "multiple_results" not in st.session_state:
    st.session_state["multiple_results"] = None


# ==========================================
# 1. 헬퍼 함수 정의
# ==========================================
def generate_sample_data():
    """기상 변수와 초미세먼지(PM2.5) 간의 관계를 모사한 120행 예제 데이터 생성"""
    np.random.seed(42)
    n = 120
    temperature = np.random.uniform(5, 35, n)
    humidity = np.random.uniform(20, 90, n)
    wind_speed = np.random.uniform(0.5, 8.0, n)
    rainfall = np.random.exponential(scale=2.0, size=n)
    rainfall[rainfall < 0.5] = 0.0  # 비가 안 오는 날 처리

    # 초미세먼지 생성 수식 (기온 높고 바람 약할 때 상승, 비 올 때 감소)
    pm25 = (
        15
        + 1.2 * temperature
        - 0.25 * humidity
        - 4.5 * wind_speed
        - 2.0 * rainfall
        + np.random.normal(0, 10, n)
    )
    pm25 = np.clip(pm25, 5, 180)  # 음수 방지 및 현실적 범위 조정

    df = pd.DataFrame(
        {
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 1),
            "rainfall": np.round(rainfall, 1),
            "pm25": np.round(pm25, 1),
        }
    )
    return df


def load_csv(uploaded_file):
    """UTF-8 및 CP949 인코딩 지원 CSV 로더"""
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")
        except Exception as e:
            st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
            return None


def calculate_metrics(y_true, y_pred, n_features):
    """모델 평가 지표 계산 (MAE, MSE, RMSE, R2, Adjusted R2)"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    if n - n_features - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = r2

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2,
    }


# ==========================================
# 2. 사이드바 구성
# ==========================================
with st.sidebar:
    st.title("🔬 실험실 메뉴")
    st.info("💡 **수업 안내**\n단계별 탭을 순서대로 진행하며 데이터와 선형회귀 모델의 관계를 탐구해보세요.")

    st.markdown("---")
    st.subheader("📚 핵심 용어 정리")
    with st.expander("용어 사전 보기"):
        st.markdown(
            """
        - **독립변수 ($X$)**: 원인이 되는 변수 (입력값)
        - **종속변수 ($y$)**: 결과가 되는 변수 (출력값)
        - **회귀계수 ($\beta$)**: $X$가 1 증가할 때 $y$의 변화량 (기울기)
        - **절편 ($\beta_0$)**: $X$가 0일 때의 $y$ 값
        - **예측값 ($\hat{y}$)**: 모델이 계산해낸 값
        - **잔차 (Residual)**: 실제값 - 예측값
        - **$R^2$**: 모델이 데이터를 설명하는 비율 (0~1)
        """
        )

    st.markdown("---")
    if st.session_state["df"] is not None:
        st.success(
            f"📂 현재 데이터 로드됨: {len(st.session_state['df'])}행, {len(st.session_state['df'].columns)}열"
        )
    else:
        st.warning("⚠️ 등록된 데이터가 없습니다.")


# ==========================================
# 3. 메인 화면 타이틀 및 탭 설정
# ==========================================
st.title("📊 CSV 데이터로 배우는 선형회귀 실험실")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1. 학습 안내",
        "2. CSV 데이터 업로드",
        "3. 데이터 탐색",
        "4. 단순선형회귀",
        "5. 다중선형회귀",
        "6. 모델 평가 및 비교",
    ]
)

# ==========================================
# Tab 1: 학습 안내
# ==========================================
with tab1:
    st.header("📘 선형회귀(Linear Regression) 핵심 개념")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. 회귀와 선형회귀")
        st.markdown(
            """
        * **회귀(Regression)**란? 어떤 연속된 수치(예: 기온, 가격, 먼지 농도 등)를 예측하는 인공지능 기법입니다.
        * **선형회귀**는 데이터 변수들 사이의 관계를 **직선 형태**로 모델링합니다.
        * **독립변수($X$)**: 예측에 사용하는 입력 변수 (예: 기온, 풍속)
        * **종속변수($y$)**: 예측하고자 하는 목표 변수 (예: 초미세먼지 농도)
        """
        )

        st.subheader("2. 단순선형회귀 vs 다중선형회귀")
        st.markdown(
            """
        * **단순선형회귀**: 독립변수 $X$가 **1개**일 때 사용합니다.
        * **다중선형회귀**: 독립변수 $X$가 **2개 이상**일 때 사용하며, 여러 원인을 함께 고려합니다.
        """
        )

    with col2:
        st.subheader("3. 회귀 방정식")
        st.markdown("**단순선형회귀 수식:**")
        st.latex(r"\hat{y} = \beta_0 + \beta_1 x")

        st.markdown("**다중선형회귀 수식:**")
        st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")

        st.markdown(
            """
        * $\hat{y}$ (y-hat): 모델이 예측한 값
        * $\beta_0$: 절편(X가 모두 0일 때의 y값)
        * $\beta_1, \beta_2, \dots$: 기울기 또는 회귀계수(각 변수의 영향을 나타냄)
        """
        )

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("4. 실제값, 예측값, 잔차")
        st.info(
            """
        * **실제값 ($y$)**: 실제 관측되거나 수집된 데이터
        * **예측값 ($\hat{y}$)**: 회귀선 상에서 모델이 예측한 값
        * **잔차 (Residual)**: $\text{실제값} - \text{예측값} = y - \hat{y}$ (직선과 데이터 점 사이의 세로 거리)
        """
        )

    with col4:
        st.subheader("5. ⚠️ 상관관계 vs 인과관계")
        st.warning(
            """
        * 두 변수 사이에 강한 상관관계(경향성)가 있다고 해서 **반드시 하나가 다른 하나의 원인(인과관계)인 것은 아닙니다.**
        * 예: 아이스크림 판매량과 물놀이 사고 수는 강한 양의 상관관계가 있지만, 진짜 원인은 '여름철 높은 기온'입니다.
        """
        )

    st.markdown("---")
    with st.expander("❓ [탐구 질문] 미리 생각해보기"):
        st.markdown(
            """
        1. 독립변수 $X$를 많이 추가할수록 항상 예측 모델의 성능이 완벽해질까요?
        2. 회귀선 위에 없는 실제 데이터 점들의 오차(잔차)를 최소화하려면 어떻게 해야 할까요?
        3. 모델이 예측한 초미세먼지 농도가 음수(-10)로 나온다면, 이를 어떻게 해석해야 할까요?
        """
        )


# ==========================================
# Tab 2: CSV 데이터 업로드
# ==========================================
with tab2:
    st.header("📂 2. CSV 데이터 업로드 및 확인")

    st.subheader("📥 예제 데이터 내려받기")
    sample_df = generate_sample_data()
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8-sig")

    col_down, col_info = st.columns([1, 2])
    with col_down:
        st.download_button(
            label="📄 미세먼지 예제 CSV 다운로드",
            data=csv_bytes,
            file_name="sample_pm25_data.csv",
            mime="text/csv",
        )
    with col_info:
        st.caption("기온(temperature), 습도(humidity), 풍속(wind_speed), 강수량(rainfall), 초미세먼지(pm25) 데이터(120행)가 포함되어 있습니다.")

    st.markdown("---")
    st.subheader("📤 내 CSV 파일 업로드")

    uploaded_file = st.file_uploader("CSV 파일을 선택하세요 (UTF-8 또는 CP949 인코딩 지원)", type=["csv"])

    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            st.session_state["df"] = df
            st.success("성공적으로 CSV 파일을 읽어왔습니다!")
    else:
        if st.session_state["df"] is None:
            st.info("파일을 업로드하거나 위의 예제 CSV 다운로드 후 다시 업로드하여 진행하세요.")
            # 기본적으로 예제 데이터를 사용 가능하도록 세팅
            st.session_state["df"] = sample_df
            st.caption("※ 파일 업로드가 없어 기본 예제 데이터세트를 세션에 로드했습니다.")

    df = st.session_state["df"]

    if df is not None:
        st.markdown("---")
        st.subheader("📋 데이터 요약 및 검증")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("전체 행 수", f"{df.shape[0]} 행")
        col_m2.metric("전체 열 수", f"{df.shape[1]} 개")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        col_m3.metric("숫자형 변수", f"{len(numeric_cols)} 개")
        col_m4.metric("문자/범주형 변수", f"{len(non_numeric_cols)} 개")

        st.write("▼ **업로드 데이터 상위 5행 미리보기**")
        st.dataframe(df.head(), use_container_width=True)

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.write("**열별 데이터 타입 및 결측값 개수**")
            info_df = pd.DataFrame(
                {
                    "데이터 타입": df.dtypes.astype(str),
                    "결측값(Null) 개수": df.isnull().sum(),
                }
            )
            st.dataframe(info_df, use_container_width=True)

        with col_info2:
            st.write("**변수 구분**")
            st.write(f"- **숫자형 변수 ($X, y$ 가능)**: {', '.join(numeric_cols) if numeric_cols else '없음'}")
            st.write(f"- **문자/범주형 변수**: {', '.join(non_numeric_cols) if non_numeric_cols else '없음'}")

            # 데이터 검증 경고 처리
            if len(numeric_cols) < 2:
                st.error("⚠️ 선형회귀 분석을 수행하려면 최소 2개 이상의 숫자형(Numeric) 열이 필요합니다.")
            elif df.shape[0] < 10:
                st.error("⚠️ 데이터 행 수가 10개 미만입니다. 모델 학습이 불가능합니다.")
            elif df.shape[0] < 30:
                st.warning("⚠️ 데이터 행 수가 30개 미만으로 적습니다. 결과 해석에 주의가 필요합니다.")

        with st.expander("❓ [탐구 질문] 데이터 업로드 단계"):
            st.markdown(
                """
            1. 준비한 데이터 중 예측하고 싶은 **종속변수($y$)**는 무엇인가요?
            2. 데이터에 결측값(Null)이 포함되어 있다면, 학습 전에 어떻게 처리해야 할까요?
            """
            )


# ==========================================
# Tab 3: 데이터 탐색 (EDA)
# ==========================================
with tab3:
    st.header("🔍 3. 데이터 탐색 및 상관관계 분석")

    df = st.session_state["df"]

    if df is None or len(df.select_dtypes(include=[np.number]).columns) < 2:
        st.warning("⚠️ 적절한 CSV 데이터를 먼저 업로드해 주세요.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        st.subheader("1. 숫자형 변수 기술통계량")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

        st.markdown("---")
        st.subheader("2. 변수별 분포 (히스토그램) & 관계 (산점도)")

        col_eda1, col_eda2 = st.columns(2)

        with col_eda1:
            selected_hist_var = st.selectbox("히스토그램으로 볼 변수 선택", numeric_cols)
            fig_hist = px.histogram(
                df,
                x=selected_hist_var,
                marginal="box",
                title=f"{selected_hist_var} 변수 분포",
                color_discrete_sequence=["#3366CC"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_eda2:
            scat_x = st.selectbox("산점도 X축 선택", numeric_cols, index=0)
            scat_y_idx = 1 if len(numeric_cols) > 1 else 0
            scat_y = st.selectbox("산점도 Y축 선택", numeric_cols, index=scat_y_idx)

            fig_scat = px.scatter(
                df,
                x=scat_x,
                y=scat_y,
                trendline="ols",
                trendline_color_override="red",
                title=f"{scat_x} vs {scat_y} 산점도",
                hover_data=df.columns,
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        # 산점도 관찰 질문 체크
        st.info(
            f"""
        💡 **[{scat_x} vs {scat_y}] 산점도 관찰 체크리스트**
        * 두 변수는 우상향(양의 관계)인가요, 우하향(음의 관계)인가요?
        * 점들이 빨간색 추세선 주변에 가깝게 모여 있나요, 넓게 퍼져 있나요?
        * 유난히 떨어진 **이상치(Outlier)** 데이터 점이 보이나요?
        * 상관관계가 관찰되더라도 인과관계로 단정할 수 있을까요?
        """
        )

        st.markdown("---")
        st.subheader("3. 상관계수 히트맵 (Correlation Heatmap)")

        corr_matrix = df[numeric_cols].corr()

        col_corr1, col_corr2 = st.columns([1, 1])

        with col_corr1:
            st.write("**상관계수 표 (Pearson Correlation)**")
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"), use_container_width=True)

        with col_corr2:
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="변수 간 상관계수 히트맵",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        st.warning(
            """
        **상관계수($r$) 이해하기**
        * **$+1.0$에 가까울수록**: 강한 양의 상관관계 (X가 늘면 Y도 증가)
        * **$-1.0$에 가까울수록**: 강한 음의 상관관계 (X가 늘면 Y는 감소)
        * **$0.0$ 근처**: 두 변수 간 선형적 관계가 거의 없음
        """
        )

        with st.expander("❓ [탐구 질문] 데이터 탐색 단계"):
            st.markdown(
                """
            1. 종속변수($y$)와 가장 강한 상관관계를 보이는 독립변수($X$)는 무엇인가요?
            2. 상관계수가 $0$에 가깝다면, 선형회귀 모델로 예측하기에 적합할까요?
            """
            )


# ==========================================
# Tab 4: 단순선형회귀
# ==========================================
with tab4:
    st.header("📈 4. 단순선형회귀 실험실")

    df = st.session_state["df"]

    if df is None or len(df.select_dtypes(include=[np.number]).columns) < 2:
        st.warning("⚠️ 적절한 CSV 데이터를 먼저 업로드해 주세요.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        col_sel1, col_sel2, col_sel3 = st.columns([2, 2, 2])
        with col_sel1:
            x_var = st.selectbox("독립변수 (X) 선택", numeric_cols, index=0, key="sim_x")
        with col_sel2:
            # y 변수는 x와 다르게 자동 지정
            default_y_idx = 1 if len(numeric_cols) > 1 else 0
            if numeric_cols[default_y_idx] == x_var and len(numeric_cols) > 2:
                default_y_idx = 2
            y_var = st.selectbox("종속변수 (y) 선택", numeric_cols, index=default_y_idx, key="sim_y")
        with col_sel3:
            test_size = st.slider("테스트 데이터 비율", 0.10, 0.40, 0.20, step=0.05)

        if x_var == y_var:
            st.error("⚠️ 독립변수(X)와 종속변수(y)는 서로 다른 변수여야 합니다.")
        else:
            # 데이터 준비 (결측치 제거)
            clean_df = df[[x_var, y_var]].dropna()

            if len(clean_df) < 10:
                st.error("⚠️ 결측치 제거 후 데이터가 10행 미만입니다. 모델을 학습할 수 없습니다.")
            else:
                X = clean_df[[x_var]]
                y = clean_df[y_var]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)

                metrics = calculate_metrics(y_test, y_pred_test, n_features=1)
                slope = model.coef_[0]
                intercept = model.intercept_
                corr_val = clean_df.corr().loc[x_var, y_var]

                # 결과를 세션에 저장
                st.session_state["simple_results"] = {
                    "x_var": x_var,
                    "y_var": y_var,
                    "slope": slope,
                    "intercept": intercept,
                    "metrics": metrics,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred_test": y_pred_test,
                    "model": model,
                }

                st.markdown("---")
                st.subheader("1. 모델 학습 결과")

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("기울기 (Slope)", f"{slope:.4f}")
                m_col2.metric("절편 (Intercept)", f"{intercept:.4f}")
                m_col3.metric("상관계수 (r)", f"{corr_val:.4f}")
                m_col4.metric("결정계수 (R²)", f"{metrics['R2']:.4f}")

                # 회귀식 표시
                sign = "+" if intercept >= 0 else "-"
                st.success(
                    f"📐 **도출된 단순선형회귀식**:  \n"
                    f"$$\text{{예측 {y_var}}} = {slope:.4f} \\times \\text{{{x_var}}} {sign} {abs(intercept):.4f}$$"
                )

                # 자동 해석
                direction = "증가" if slope > 0 else "감소"
                st.info(
                    f"💡 **기울기 해석**: **[{x_var}]** 변수가 1 단위 증가할 때, **[{y_var}]** 예측값은 평균적으로 약 **{abs(slope):.4f}** 만큼 **{direction}**합니다. (단, 이는 경향성이며 인과관계를 단정하지 않습니다.)"
                )

                st.markdown("---")
                st.subheader("2. 산점도 및 회귀선 / 잔차 시각화")

                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    # 학습/테스트 구분 시각화
                    fig_sim = go.Figure()

                    fig_sim.add_trace(
                        go.Scatter(
                            x=X_train[x_var],
                            y=y_train,
                            mode="markers",
                            name="학습 데이터",
                            marker=dict(color="blue", opacity=0.6),
                        )
                    )
                    fig_sim.add_trace(
                        go.Scatter(
                            x=X_test[x_var],
                            y=y_test,
                            mode="markers",
                            name="테스트 데이터",
                            marker=dict(color="orange", size=8),
                        )
                    )

                    # 회귀선 생성
                    x_range = np.linspace(clean_df[x_var].min(), clean_df[x_var].max(), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))

                    fig_sim.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode="lines",
                            name="선형회귀선",
                            line=dict(color="red", width=2),
                        )
                    )

                    fig_sim.update_layout(
                        title="학습/테스트 데이터와 회귀선",
                        xaxis_title=x_var,
                        yaxis_title=y_var,
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)

                with col_p2:
                    # 잔차 시각화 (테스트 데이터 기준)
                    fig_res = go.Figure()

                    # 데이터 점
                    fig_res.add_trace(
                        go.Scatter(
                            x=X_test[x_var],
                            y=y_test,
                            mode="markers",
                            name="실제값",
                            marker=dict(color="orange", size=8),
                        )
                    )

                    # 예측값 점
                    fig_res.add_trace(
                        go.Scatter(
                            x=X_test[x_var],
                            y=y_pred_test,
                            mode="markers",
                            name="예측값",
                            marker=dict(color="red", symbol="x", size=8),
                        )
                    )

                    # 잔차 연결선 (실제값과 예측값 사이)
                    for x_val, y_t, y_p in zip(X_test[x_var], y_test, y_pred_test):
                        fig_res.add_shape(
                            type="line",
                            x0=x_val,
                            y0=y_t,
                            x1=x_val,
                            y1=y_p,
                            line=dict(color="gray", width=1, dash="dot"),
                        )

                    fig_res.update_layout(
                        title="테스트 데이터의 잔차(Residual) 시각화",
                        xaxis_title=x_var,
                        yaxis_title=y_var,
                    )
                    st.plotly_chart(fig_res, use_container_width=True)

                st.markdown("---")
                st.subheader("3. 🎯 새로운 X값으로 y 예측해보기")

                x_min_val = float(clean_df[x_var].min())
                x_max_val = float(clean_df[x_var].max())
                x_mean_val = float(clean_df[x_var].mean())

                user_x_val = st.number_input(
                    f"새로운 [{x_var}] 값 입력 (기존 범위: {x_min_val:.1f} ~ {x_max_val:.1f})",
                    value=round(x_mean_val, 1),
                )

                predicted_y_val = model.predict([[user_x_val]])[0]

                st.write(
                    f"👉 **예측 결과**: [{x_var}] = **{user_x_val}** 일 때, 예측된 [{y_var}] = **{predicted_y_val:.2f}**"
                )

                if predicted_y_val < 0:
                    st.warning(
                        "⚠️ **참고**: 예측된 결과가 **음수(-)**입니다. 미세먼지 농도나 강수량처럼 물리적으로 음수가 될 수 없는 변수인 경우, 이는 데이터 범위 한계나 단순 선형 모델의 한계점입니다."
                    )

                st.caption(
                    "📌 *이 값은 데이터에서 학습한 선형적인 경향을 이용한 예측값이며 실제값과 다를 수 있습니다.*"
                )

                with st.expander("❓ [탐구 질문] 단순선형회귀 단계"):
                    st.markdown(
                        """
                    1. 회귀선이 모든 데이터 점의 한가운데를 지나가나요?
                    2. 잔차가 양수(+)라는 것은 실제값이 예측값보다 크다는 뜻일까요, 작다는 뜻일까요?
                    """
                    )


# ==========================================
# Tab 5: 다중선형회귀
# ==========================================
with tab5:
    st.header("📊 5. 다중선형회귀 실험실")

    df = st.session_state["df"]

    if df is None or len(df.select_dtypes(include=[np.number]).columns) < 3:
        st.warning("⚠️ 다중선형회귀를 진행하려면 최소 3개 이상의 숫자형 열이 필요합니다.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        col_m1, col_m2 = st.columns([2, 2])

        with col_m1:
            y_var_multi = st.selectbox("종속변수 (y) 선택", numeric_cols, index=len(numeric_cols) - 1, key="mul_y")

        # y 변수를 제외한 X 후보
        x_candidates = [c for c in numeric_cols if c != y_var_multi]

        with col_m2:
            x_vars_multi = st.multiselect(
                "독립변수들 (X) 선택 (최소 2개 이상)",
                x_candidates,
                default=x_candidates[:2] if len(x_candidates) >= 2 else x_candidates,
            )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_std = st.checkbox("변수 표준화 (StandardScaler) 적용", value=False)
            st.caption("※ 표준화를 적용하면 변수 간 단위 차이에 따른 회귀계수 비교가 용이해집니다.")
        with col_opt2:
            test_size_multi = st.slider("테스트 데이터 비율 (다중)", 0.10, 0.40, 0.20, step=0.05, key="mul_test_size")

        if len(x_vars_multi) < 2:
            st.error("⚠️ 다중선형회귀를 실행하려면 독립변수(X)를 최소 2개 이상 선택해야 합니다.")
        else:
            clean_df_multi = df[x_vars_multi + [y_var_multi]].dropna()

            if len(clean_df_multi) < 10:
                st.error("⚠️ 결측치 제거 후 데이터가 10행 미만입니다.")
            else:
                X_m = clean_df_multi[x_vars_multi]
                y_m = clean_df_multi[y_var_multi]

                X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                    X_m, y_m, test_size=test_size_multi, random_state=42
                )

                if use_std:
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("regressor", LinearRegression()),
                    ])
                    pipeline.fit(X_train_m, y_train_m)
                    reg_model = pipeline.named_steps["regressor"]
                    y_pred_m = pipeline.predict(X_test_m)
                    coefs = reg_model.coef_
                    intercept_m = reg_model.intercept_
                else:
                    pipeline = LinearRegression()
                    pipeline.fit(X_train_m, y_train_m)
                    coefs = pipeline.coef_
                    intercept_m = pipeline.intercept_
                    y_pred_m = pipeline.predict(X_test_m)

                metrics_m = calculate_metrics(y_test_m, y_pred_m, n_features=len(x_vars_multi))

                # 세션에 기록
                st.session_state["multiple_results"] = {
                    "x_vars": x_vars_multi,
                    "y_var": y_var_multi,
                    "coefs": coefs,
                    "intercept": intercept_m,
                    "metrics": metrics_m,
                    "X_test": X_test_m,
                    "y_test": y_test_m,
                    "y_pred_test": y_pred_m,
                    "use_std": use_std,
                    "pipeline": pipeline,
                }

                st.markdown("---")
                st.subheader("1. 다중선형회귀 평가 결과")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("결정계수 (R²)", f"{metrics_m['R2']:.4f}")
                mc2.metric("조정된 R² (Adj R²)", f"{metrics_m['Adj_R2']:.4f}")
                mc3.metric("MAE", f"{metrics_m['MAE']:.4f}")
                mc4.metric("RMSE", f"{metrics_m['RMSE']:.4f}")

                # 다중선형회귀 식 조립
                eq_terms = [f"({c:.4f} \\times \\text{{{var}}})" for c, var in zip(coefs, x_vars_multi)]
                eq_str = " + ".join(eq_terms)
                sign_m = "+" if intercept_m >= 0 else "-"
                st.success(
                    f"📐 **도출된 다중선형회귀식**:  \n"
                    f"$$\text{{예측 {y_var_multi}}} = {eq_str} {sign_m} {abs(intercept_m):.4f}$$"
                )

                st.markdown("---")
                st.subheader("2. 변수별 회귀계수(Coefficients) 비교")

                col_c1, col_c2 = st.columns(2)

                coef_df = pd.DataFrame({"독립변수": x_vars_multi, "회귀계수": coefs})

                with col_c1:
                    st.write("**변수별 회귀계수 표**")
                    st.dataframe(coef_df, use_container_width=True)

                with col_c2:
                    fig_coef = px.bar(
                        coef_df,
                        x="독립변수",
                        y="회귀계수",
                        color="회귀계수",
                        color_continuous_scale="Viridis",
                        title="독립변수별 회귀계수 크기",
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)

                st.warning(
                    """
                ⚠️ **회귀계수 해석 시 주의사항**
                * 다중선형회귀의 회귀계수는 **"다른 입력 변수들이 일정하다고 가정했을 때"** 해당 변수가 1만큼 변할 때의 예측값 변화를 의미합니다.
                * **단위(Unit) 차이 유의**: 변수마다 단위(예: 기온 ℃ vs 강수량 mm)가 다르면 회귀계수의 단순 크기만으로 어떤 변수가 더 중요한지 직접 비교하기 어렵습니다. (표준화를 적용하면 비교가 용이해집니다.)
                """
                )

                st.markdown("---")
                st.subheader("3. 🎯 새로운 입력값으로 종속변수 예측하기")

                user_inputs = {}
                col_in = st.columns(len(x_vars_multi))

                for idx, var in enumerate(x_vars_multi):
                    with col_in[idx]:
                        default_val = float(clean_df_multi[var].mean())
                        user_inputs[var] = st.number_input(
                            f"[{var}] 입력",
                            value=round(default_val, 1),
                            key=f"input_multi_{var}",
                        )

                # 예측
                input_df = pd.DataFrame([user_inputs])
                if use_std:
                    pred_multi_val = pipeline.predict(input_df)[0]
                else:
                    pred_multi_val = pipeline.predict(input_df)[0]

                st.write(
                    f"👉 **다중 모델 예측 결과**: 선택된 조건에서의 예측 [{y_var_multi}] = **{pred_multi_val:.2f}**"
                )

                with st.expander("❓ [탐구 질문] 다중선형회귀 단계"):
                    st.markdown(
                        """
                    1. 단순선형회귀 때보다 독립변수를 추가했을 때 $R^2$ 값은 어떻게 변했나요?
                    2. 독립변수를 계속 많이 추가하면 무조건 좋은 모델이 될까요? '조정된 $R^2$'의 역할을 생각해봅시다.
                    """
                    )


# ==========================================
# Tab 6: 모델 평가 및 비교
# ==========================================
with tab6:
    st.header("⚖️ 6. 모델 평가 및 비교")

    simple_res = st.session_state["simple_results"]
    multi_res = st.session_state["multiple_results"]

    if simple_res is None or multi_res is None:
        st.warning(
            "⚠️ 단순선형회귀(Tab 4)와 다중선형회귀(Tab 5) 모델을 모두 학습시킨 후에 비교할 수 있습니다."
        )
    else:
        st.subheader("1. 단순선형회귀 vs 다중선형회귀 성능 표 비교")

        comp_data = {
            "구분": ["단순선형회귀", "다중선형회귀"],
            "종속변수 (y)": [simple_res["y_var"], multi_res["y_var"]],
            "독립변수 (X)": [
                simple_res["x_var"],
                ", ".join(multi_res["x_vars"]),
            ],
            "R² (결정계수)": [
                simple_res["metrics"]["R2"],
                multi_res["metrics"]["R2"],
            ],
            "조정된 R²": [
                simple_res["metrics"]["Adj_R2"],
                multi_res["metrics"]["Adj_R2"],
            ],
            "MAE": [simple_res["metrics"]["MAE"], multi_res["metrics"]["MAE"]],
            "MSE": [simple_res["metrics"]["MSE"], multi_res["metrics"]["MSE"]],
            "RMSE": [simple_res["metrics"]["RMSE"], multi_res["metrics"]["RMSE"]],
        }

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.subheader("2. 모델 평가 지표의 의미")

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.info(
                """
            **MAE (Mean Absolute Error)**
            * 실제값과 예측값 차이의 **절댓값 평균**
            * 직관적으로 평균 오차가 얼마인지 보여줌
            """
            )
        with col_e2:
            st.info(
                """
            **MSE & RMSE**
            * **MSE**: 오차 제곱의 평균 (큰 오차에 가중치)
            * **RMSE**: MSE에 제곱근을 씌워 원래 단위로 복원한 지표
            """
            )
        with col_e3:
            st.info(
                """
            **R² & 조정된 R²**
            * **R²**: 모델이 $y$의 변동성을 설명하는 비율
            * **조정된 R²**: 변수를 무작정 늘리는 것에 벌점을 준 보정 지표
            """
            )

        st.markdown("---")
        st.subheader("3. 시각적 잔차 및 예측 정확도 진단")

        # 비교 대상 모델 선택
        target_model = st.radio("진단할 모델 선택", ["단순선형회귀", "다중선형회귀"], horizontal=True)

        if target_model == "단순선형회귀":
            res_data = simple_res
        else:
            res_data = multi_res

        y_test_target = res_data["y_test"]
        y_pred_target = res_data["y_pred_test"]
        residuals = y_test_target - y_pred_target

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # 1. 실제값 vs 예측값 산점도
            fig_act_pred = go.Figure()
            fig_act_pred.add_trace(
                go.Scatter(
                    x=y_test_target,
                    y=y_pred_target,
                    mode="markers",
                    name="데이터점",
                    marker=dict(color="purple", opacity=0.7),
                )
            )

            # 기준선 (y = x)
            min_val = min(y_test_target.min(), y_pred_target.min())
            max_val = max(y_test_target.max(), y_pred_target.max())
            fig_act_pred.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="이상적 대각선 (y=x)",
                    line=dict(color="black", dash="dash"),
                )
            )

            fig_act_pred.update_layout(
                title="실제값 vs 예측값 산점도",
                xaxis_title="실제값 (Actual)",
                yaxis_title="예측값 (Predicted)",
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with col_g2:
            # 2. 잔차 산점도 (예측값 vs 잔차)
            fig_res_scat = go.Figure()
            fig_res_scat.add_trace(
                go.Scatter(
                    x=y_pred_target,
                    y=residuals,
                    mode="markers",
                    name="잔차",
                    marker=dict(color="teal", opacity=0.7),
                )
            )
            # 기준선 0
            fig_res_scat.add_hline(y=0, line_dash="dash", line_color="red")

            fig_res_scat.update_layout(
                title="잔차 산점도 (예측값 vs 잔차)",
                xaxis_title="예측값 (Predicted)",
                yaxis_title="잔차 (Residual)",
            )
            st.plotly_chart(fig_res_scat, use_container_width=True)

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            # 3. 잔차 히스토그램
            fig_res_hist = px.histogram(
                x=residuals,
                nbins=15,
                title="잔차 분포 히스토그램",
                labels={"x": "잔차 (Residual)"},
                color_discrete_sequence=["#2E8B57"],
            )
            st.plotly_chart(fig_res_hist, use_container_width=True)

        with col_g4:
            st.markdown("#### 💡 잔차 진단 가이드")
            st.success(
                """
            * **점들이 대각선 기준선 가까이에 있을수록** 예측 정밀도가 높습니다.
            * **잔차가 0을 중심으로 무작위로 무늬 없이 퍼져 있으면** 선형회귀 모델이 적절함을 의미합니다.
            * 만약 잔차 산점도에 **U자 형태나 곡선 패턴**이 보인다면, 변수 간 관계가 비선형(Non-linear)일 가능성이 높습니다.
            """
            )

        st.markdown("---")
        st.subheader("4. 🧪 최종 모델 선택 가이드")
        st.info(
            """
        * 무조건 $R^2$가 더 높다고 해서 '다중선형회귀'를 선택하는 것은 바람직하지 않을 수 있습니다.
        * **복잡성과 설명력의 트레이드오프(Trade-off)**: 성능 향상 폭(RMSE 감소)이 적다면, 구조가 단순하고 설명하기 쉬운 **단순선형회귀**가 더 유용한 모델일 수 있습니다.
        """
        )

        with st.expander("❓ [탐구 질문] 모델 평가 및 비교 단계"):
            st.markdown(
                """
            1. 다중선형회귀로 가면서 $R^2$는 얼마나 증가했고, RMSE는 얼마나 감소했나요?
            2. 이 모델의 예측 결과를 실제 미세먼지 예보 방송에 그대로 활용해도 될까요? 한계점은 무엇일까요?
            """
            )
