import torch
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from model import BidirectionalLSTMModel


def load_model_and_scaler(model_path, scaler_path, num_features=10, hidden_size=128, num_layers=3):
    # 모델 로드
    model = BidirectionalLSTMModel(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 스케일러 로드
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_next_month(model, scaler, input_data):
    # 텐서 변환
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # (1, 36, 10)

    # 예측
    with torch.no_grad():
        prediction = model(input_tensor)  # (1, 10)

    # 스케일 복원
    prediction_original = scaler.inverse_transform(prediction.numpy())

    return prediction_original[0]

# 향후 6개월 예측 (연속 예측)
def predict_next_n_months(model, scaler, initial_data, n_months=6):
    predictions = []
    current_input = initial_data.copy()

    for i in range(n_months):
        # 예측
        input_tensor = torch.FloatTensor(current_input).unsqueeze(0)

        with torch.no_grad():
            pred = model(input_tensor)

        pred_numpy = pred.numpy()

        # 스케일 복원하여 저장
        pred_original = scaler.inverse_transform(pred_numpy)
        predictions.append(pred_original[0])

        # 슬라이딩 윈도우: 가장 오래된 달 제거, 새 예측 추가
        current_input = np.vstack([current_input[1:], pred_numpy])

    return np.array(predictions)


def main():
    print("=" * 70)
    print("TIOBE 프로그래밍 언어 트렌드 예측")
    print("=" * 70)

    # 경로 설정
    data_dir = Path("/Users/parkjuyong/Desktop/4-1/CareerRoute/ml/trend_tech_ml/data")
    model_path = data_dir / "results" / "best_model.pth"
    scaler_path = data_dir / "scaler.pkl"
    data_path = data_dir / "scaled_tiobe_data.csv"

    # 언어 이름
    languages = ['Python', 'Java', 'C++', 'C#', 'C', 'JavaScript', 'SQL', 'R', 'Visual Basic', 'Perl']

    # 모델 로드
    print("\n모델 로딩...")
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    print(f"    모델 로드 완료: {model_path}")

    # 최신 36개월 데이터 로드
    print("\n최신 데이터 로딩...")
    df = pd.read_csv(data_path)
    latest_36_months = df.iloc[-36:, 1:].values  # 마지막 36개월
    latest_date = df.iloc[-1, 0]
    print(f"    데이터 로드 완료: 최신 날짜 = {latest_date}")
    print(f"    입력 데이터: 최근 36개월 ({df.iloc[-36, 0]} ~ {latest_date})")

    # 다음 달 예측
    print("\n다음 달 예측 중...")
    next_month_pred = predict_next_month(model, scaler, latest_36_months)

    print("\n" + "=" * 70)
    print("2026년 1월 예측 결과")
    print("=" * 70)
    print(f"{'언어':<20} {'예측값':>10} {'순위':>10}")
    print("-" * 70)

    # 순위 계산
    sorted_indices = np.argsort(next_month_pred)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"{languages[idx]:<20} {next_month_pred[idx]:>9.2f}% {rank:>10}")

    # 향후 6개월 예측
    print("\n\n향후 6개월 예측 중...")
    future_6_months = predict_next_n_months(model, scaler, latest_36_months, n_months=6)

    print("\n" + "=" * 70)
    print("2026년 1월 ~ 6월 예측 결과")
    print("=" * 70)

    # 월별 출력
    for month_idx, month_pred in enumerate(future_6_months, 1):
        print(f"\n【 2026년 {month_idx}월 】")
        print(f"{'언어':<20} {'예측값':>10}")
        print("-" * 35)

        sorted_indices = np.argsort(month_pred)[::-1]
        for idx in sorted_indices[:5]:  # 상위 5개만
            print(f"{languages[idx]:<20} {month_pred[idx]:>9.2f}%")

    # 5. 트렌드 분석
    print("\n\n" + "=" * 70)
    print("6개월 트렌드 분석 (1월 대비 6월 변화)")
    print("=" * 70)
    print(f"{'언어':<20} {'1월':>10} {'6월':>10} {'변화':>10}")
    print("-" * 70)

    change = future_6_months[-1] - future_6_months[0]
    sorted_by_change = np.argsort(change)[::-1]

    for idx in sorted_by_change:
        jan_val = future_6_months[0][idx]
        jun_val = future_6_months[-1][idx]
        change_val = change[idx]
        trend = "(상승)" if change_val > 0.1 else "(하락)" if change_val < -0.1 else "(유지)"
        print(f"{languages[idx]:<20} {jan_val:>9.2f}% {jun_val:>9.2f}% {change_val:>+9.2f}% {trend}")

    print("\n" + "=" * 70)
    print("예측 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
