# Agent Harness Framework

## Giới thiệu
**Agent Harness Framework** là một nền tảng hạ tầng hướng sản xuất dành cho các AI agent, hỗ trợ xây dựng, kiểm thử, đánh giá và triển khai agent một cách an toàn, kiểm soát và mở rộng.

## Tính năng nổi bật
- Quản lý vòng đời agent: khởi tạo, lập kế hoạch, thực thi, đánh giá.
- Hỗ trợ nhiều mô hình AI (OpenAI, Anthropic, v.v.) và tích hợp dễ dàng.
- Hệ thống skill/plugin mở rộng linh hoạt.
- Quản lý ngân sách, quyền hạn, kiểm soát rủi ro khi gọi tool.
- Lưu vết, đánh giá chất lượng, kiểm thử tự động.
- API service (FastAPI) và CLI tiện dụng.

## Yêu cầu hệ thống
- Python >= 3.11

## Cài đặt
```bash
git clone https://github.com/your-org/agent-harness-framework.git
cd agent-harness-framework
pip install .
# Hoặc cài thêm các tuỳ chọn:
pip install .[llm,service,sandbox,test]
```

## Sử dụng nhanh
Chạy agent với mục tiêu cụ thể:
```bash
python -m agent_harness.cli run "Tóm tắt tài liệu PDF"
```
Hoặc dùng CLI:
```bash
python -m agent_harness.cli --help
```

## Cấu trúc thư mục
- `src/agent_harness/` - Mã nguồn chính
- `skills/` - Ví dụ skill mở rộng
- `configs/` - Cấu hình mẫu
- `storage/` - Lưu trữ kết quả, trace, workspace
- `tests/` - Unit test

## Đóng góp
Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## Giấy phép
Dự án phát hành theo giấy phép MIT.

---

# Agent Harness Framework (English)

## Overview
**Agent Harness Framework** is a production-oriented infrastructure for AI agents, supporting safe, controlled, and extensible agent development, evaluation, and deployment.

## Features
- Agent lifecycle management: initialization, planning, execution, evaluation
- Multi-model support (OpenAI, Anthropic, etc.) and easy integration
- Flexible skill/plugin system
- Budget, permission, and risk control for tool calls
- Tracing, quality evaluation, automated testing
- API service (FastAPI) and handy CLI

## Requirements
- Python >= 3.11

## Installation
```bash
git clone https://github.com/your-org/agent-harness-framework.git
cd agent-harness-framework
pip install .
# Or with extras:
pip install .[llm,service,sandbox,test]
```

## Quick Start
Run an agent with a specific goal:
```bash
python -m agent_harness.cli run "Summarize a PDF document"
```
Or use the CLI:
```bash
python -m agent_harness.cli --help
```

## Project Structure
- `src/agent_harness/` - Main source code
- `skills/` - Example extension skills
- `configs/` - Sample configurations
- `storage/` - Results, traces, workspaces
- `tests/` - Unit tests

## Contributing
Contributions are welcome! Please open an issue or pull request.

## License
This project is licensed under the MIT License.
