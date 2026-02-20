# Skills Usage Guide

## Vị trí skills

- `.agents/skills/trading-data-pipeline`
- `.agents/skills/trading-feature-pipeline`

## Skill 1: `trading-data-pipeline`

Mục đích:
- Tải dữ liệu BTCUSDT 15m theo `configs/config.yaml`
- Validate dữ liệu raw
- Báo trạng thái `artifacts/raw/BTCUSDT_15m.csv` và `artifacts/reports/data_quality.json`

Prompt mẫu:
- `Use $trading-data-pipeline để tải lại dữ liệu mới nhất trong khoảng ngày hiện tại của config và validate giúp tôi.`
- `Use $trading-data-pipeline để kiểm tra vì sao data download bị lỗi và báo rõ bước lỗi.`
- `Use $trading-data-pipeline, sau khi chạy xong hãy tóm tắt số dòng, timestamp đầu/cuối và kết quả validation.`

## Skill 2: `trading-feature-pipeline`

Mục đích:
- Build feature Phase 2 (core + structure)
- Kiểm tra output parquet có đúng schema/rỗng/null bất thường không
- Đảm bảo logic no-lookahead với swing confirmation delay

Prompt mẫu:
- `Use $trading-feature-pipeline để build lại features và báo shape + columns của 2 file output.`
- `Use $trading-feature-pipeline để kiểm tra các cột atr14, ret1, vol_ratio có NaN bất thường không.`
- `Use $trading-feature-pipeline để debug lỗi feature engineering hiện tại và đề xuất cách sửa.`

## Quy trình dùng hằng ngày

1. Chạy data pipeline:
   - `Use $trading-data-pipeline để refresh data và validate giúp tôi.`
2. Chạy feature pipeline:
   - `Use $trading-feature-pipeline để build features và kiểm tra nhanh output.`
3. Nếu có lỗi:
   - Gọi lại đúng skill tương ứng với lỗi (data lỗi gọi skill 1, feature lỗi gọi skill 2).

## Mẹo

- Nói rõ mục tiêu + output muốn nhận (ví dụ: “cho tôi bảng tóm tắt số dòng và cột bị NaN”).
- Nếu cần chạy theo mốc thời gian khác, nhắc rõ muốn đổi trường nào trong `configs/config.yaml`.
