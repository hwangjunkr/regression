# -*- coding: utf-8 -*-
"""
단순선형회귀 (Simple Linear Regression)  —  y = a*x + b
편미분으로 유도한 최소제곱법 공식과 경사하강법을 함께 구현한 것.

[사용법]
  1) 예시 데이터로 바로 실행
       python linear_regression.py

  2) 내 CSV 파일로 실행  (첫 두 열을 x, y로 사용. 헤더 있어도 됨)
       python linear_regression.py mydata.csv

  3) 다른 코드에서 함수만 가져다 쓰기
       from linear_regression import fit
       result = fit(x_list, y_list)
       print(result["a"], result["b"], result["r2"])

[필요 라이브러리]  numpy, matplotlib
       pip install numpy matplotlib
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로 저장 (터미널/서버에서도 동작)
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


# =========================================================
# 0. 한글 폰트 설정
#    - rcParams는 없는 폰트를 넣어도 에러가 안 나므로,
#      실제 설치된 폰트 목록에서 확인 후 지정해야 한다.
# =========================================================
def setup_korean_font():
    """한글 폰트를 찾으면 적용하고 True, 없으면 False 반환."""
    plt.rcParams["axes.unicode_minus"] = False
    # 로그축 눈금은 mathtext로 그려지는데, 한글 폰트에는 수식 기호가
    # 없는 경우가 많다. 수식용 폰트만 별도로 고정해 깨짐을 막는다.
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "NanumBarunGothic", "Noto Sans CJK KR", "NanumSquare"]
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            return True
    return False


KOREAN_OK = setup_korean_font()


def L(ko, en):
    """한글 폰트가 없으면 영문 라벨로 대체 (네모 깨짐 방지)."""
    return ko if KOREAN_OK else en


# =========================================================
# 1. 최소제곱법 (Closed-form Least Squares)
#
#    편미분으로 유도한 결과:
#        a = Σ xi(yi - Ȳ) / Σ xi(xi - X̄)
#    이는 아래 중심화(centered) 형태와 수학적으로 동일하며,
#    수치적으로 더 안정적이므로 이쪽을 사용한다.
#        a = Σ(xi - X̄)(yi - Ȳ) / Σ(xi - X̄)²
#        b = Ȳ - a·X̄
# =========================================================
def least_squares_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_bar, y_bar = x.mean(), y.mean()
    denominator = np.sum((x - x_bar) ** 2)

    if np.isclose(denominator, 0.0):
        raise ValueError(
            "x값이 모두 동일합니다. 기울기를 정의할 수 없습니다 "
            "(수직선은 y=ax+b로 표현 불가)."
        )

    a = np.sum((x - x_bar) * (y - y_bar)) / denominator
    b = y_bar - a * x_bar
    return float(a), float(b)


# =========================================================
# 2. 경사하강법 (Gradient Descent)
#
#    오차함수  E = mean((y - (a·x + b))²)
#
#    [중요] 원본 스케일 그대로 돌리면 x가 클 때 발산한다.
#    따라서 내부에서 표준화(z-score) 후 학습하고,
#    마지막에 계수를 원래 스케일로 되돌린다.
# =========================================================
def gradient_descent_fit(x, y, lr=0.05, epochs=3000, tol=1e-12, verbose=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    mx, sx = x.mean(), x.std()
    my, sy = y.mean(), y.std()

    if np.isclose(sx, 0.0):
        raise ValueError("x값이 모두 동일합니다. 기울기를 정의할 수 없습니다.")
    if np.isclose(sy, 0.0):
        # y가 상수면 기울기 0, 절편은 그 상수값
        return 0.0, float(my), [0.0]

    xs = (x - mx) / sx          # 표준화된 x
    ys = (y - my) / sy          # 표준화된 y

    a_s, b_s = 0.0, 0.0
    history = []
    prev_mse = float("inf")

    for epoch in range(1, epochs + 1):
        error = ys - (a_s * xs + b_s)

        grad_a = (-2.0 / n) * np.sum(xs * error)
        grad_b = (-2.0 / n) * np.sum(error)

        a_s -= lr * grad_a
        b_s -= lr * grad_b

        mse = float(np.mean(error ** 2))
        history.append(mse)

        if verbose and epoch % 500 == 0:
            print(f"    epoch {epoch:5d} | MSE(표준화){mse:12.8f}")

        # 수렴 판정: 개선폭이 tol 미만이면 조기 종료
        if abs(prev_mse - mse) < tol:
            if verbose:
                print(f"    epoch {epoch}에서 수렴하여 조기 종료")
            break
        prev_mse = mse

    # 원래 스케일로 계수 환원
    a = (sy / sx) * a_s
    b = my + sy * b_s - a * mx
    return float(a), float(b), history


# =========================================================
# 3. 성능 지표
# =========================================================
def r_squared(x, y, a, b):
    """결정계수 R² = 1 - SS_res/SS_tot"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ss_res = np.sum((y - (a * x + b)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if np.isclose(ss_tot, 0.0):
        return float("nan")  # y가 상수면 R²는 정의되지 않음
    return float(1 - ss_res / ss_tot)


def rmse(x, y, a, b):
    """평균제곱근오차 — y와 단위가 같아 해석하기 쉬움"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean((y - (a * x + b)) ** 2)))


# =========================================================
# 4. 통합 인터페이스
# =========================================================
def fit(x, y, method="least_squares", **kwargs):
    """
    method: "least_squares" (기본, 정확해) 또는 "gradient_descent"
    반환: dict(a, b, r2, rmse, equation, history)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"x와 y의 개수가 다릅니다: {x.shape} vs {y.shape}")
    if len(x) < 2:
        raise ValueError("데이터가 2개 미만이면 직선을 결정할 수 없습니다.")

    history = None
    if method == "least_squares":
        a, b = least_squares_fit(x, y)
    elif method == "gradient_descent":
        a, b, history = gradient_descent_fit(x, y, **kwargs)
    else:
        raise ValueError("method는 'least_squares' 또는 'gradient_descent'")

    sign = "+" if b >= 0 else "-"
    return {
        "a": a,
        "b": b,
        "r2": r_squared(x, y, a, b),
        "rmse": rmse(x, y, a, b),
        "equation": f"y = {a:.6f}x {sign} {abs(b):.6f}",
        "history": history,
    }


def predict(x_new, a, b):
    """학습된 계수로 새 x에 대한 y 예측"""
    return np.asarray(x_new, dtype=float) * a + b


# =========================================================
# 5. 시각화
# =========================================================
def plot_result(x, y, a, b, history=None, title=None, save_path="regression.png"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_pred = a * x + b

    ncols = 2 if history else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))
    ax = axes[0] if history else axes

    # --- 회귀직선 + 잔차 ---
    ax.scatter(x, y, color="black", s=35, zorder=3,
               label=L("관측 데이터", "Observed"))
    for xi, yi, ypi in zip(x, y, y_pred):
        ax.plot([xi, xi], [yi, ypi], color="magenta", lw=1, alpha=0.6, zorder=2)

    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, a * xl + b, color="red", lw=2, zorder=4,
            label=f"y = {a:.3f}x + {b:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or L("회귀 결과", "Regression Result"))
    ax.legend()
    ax.grid(alpha=0.3)

    # --- 손실 곡선 ---
    if history:
        ax2 = axes[1]
        ax2.plot(history, color="steelblue", lw=1.5)
        ax2.set_xlabel(L("반복 횟수 (epoch)", "Epoch"))
        ax2.set_ylabel(L("오차 (MSE, 표준화)", "MSE (standardized)"))
        ax2.set_title(L("경사하강법 수렴 과정", "Gradient Descent Convergence"))
        ax2.set_yscale("log")
        # 로그축 기본 눈금은 mathtext(10⁻³ 형태)로 그려지는데, 한글 폰트에
        # 유니코드 마이너스가 없으면 깨진다. 일반 텍스트 포맷터로 대체.
        ax2.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-"))
        )
        ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  그래프 저장: {save_path}")


# =========================================================
# 6. CSV 로딩
# =========================================================
def load_csv(path):
    """첫 두 열을 x, y로 읽는다. 헤더 유무 자동 판별."""
    raw = np.genfromtxt(path, delimiter=",", dtype=str,
                        encoding="utf-8-sig", autostrip=True)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 2:
        raise ValueError("CSV에 열이 2개 이상 있어야 합니다 (x, y).")

    # 첫 행이 숫자로 변환 안 되면 헤더로 간주하고 건너뜀
    try:
        float(raw[0, 0])
        start = 0
    except ValueError:
        start = 1

    data = raw[start:, :2].astype(float)
    return data[:, 0], data[:, 1]


# =========================================================
# 7. 실행
# =========================================================
def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        x, y = load_csv(path)
        print(f"'{path}'에서 {len(x)}개 데이터를 읽었습니다.\n")
    else:
        print("[예시 데이터로 실행합니다. CSV를 쓰려면: "
              "python linear_regression.py mydata.csv]\n")
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 3 + rng.normal(0, 2, size=x.shape)

    print("=" * 52)
    print("[1] 최소제곱법 — 정확해 (한 번에 계산)")
    print("=" * 52)
    r1 = fit(x, y, method="least_squares")
    print(f"  {r1['equation']}")
    print(f"  R²   = {r1['r2']:.6f}")
    print(f"  RMSE = {r1['rmse']:.6f}")
    plot_result(x, y, r1["a"], r1["b"],
                title=L("최소제곱법", "Least Squares"),
                save_path="result_least_squares.png")

    print("\n" + "=" * 52)
    print("[2] 경사하강법 — 반복 학습")
    print("=" * 52)
    r2 = fit(x, y, method="gradient_descent", verbose=True)
    print(f"  {r2['equation']}")
    print(f"  R²   = {r2['r2']:.6f}")
    print(f"  RMSE = {r2['rmse']:.6f}")
    plot_result(x, y, r2["a"], r2["b"], history=r2["history"],
                title=L("경사하강법", "Gradient Descent"),
                save_path="result_gradient_descent.png")

    print("\n" + "=" * 52)
    print("[검증] 두 방법의 계수 차이")
    print("=" * 52)
    print(f"  Δa = {abs(r1['a'] - r2['a']):.3e}")
    print(f"  Δb = {abs(r1['b'] - r2['b']):.3e}")
    print("  (경사하강법이 최소제곱법의 해로 수렴했다면 0에 가까워야 함)")


if __name__ == "__main__":
    main()
