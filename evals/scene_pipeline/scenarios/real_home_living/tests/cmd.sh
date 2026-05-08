# 默认使用 Qwen3-VL-235B-A22B-Instruct
python -m pytest examples/real_home_living/tests/test_task_planning.py -v

# 使用 Qwen3-235B-A22B-Instruct-2507
TEST_MODEL=Qwen3-235B-A22B-Instruct-2507 \
TEST_BASE_URL=http://120.48.76.60:5014/v1 \
TEST_API_KEY=BbuxnkdcbvLRLJOedui6CEs8J7rkZXAfWeGtKTh4jEw \
python -m pytest examples/real_home_living/tests/test_task_planning.py -v

# 运行全部 Layer 1 测试
python -m pytest examples/real_home_living/tests/ -v --ignore=examples/real_home_living/tests/test_e2e.py

# 使用 gpt-4.1 模型运行全部 Layer 1 测试
TEST_MODEL=gpt-4.1 \
TEST_BASE_URL=https://api.chatanywhere.tech/v1 \
TEST_API_KEY=sk-OltbgRT93Ua8BGX85EgZWNV02t9WtAJt4A6xoRtAIak7IyNE \
python -m pytest examples/real_home_living/tests/ -v --ignore=examples/real_home_living/tests/test_e2e.py
# python -m pytest examples/real_home_living/tests/test_task_planning.py::TestFetchAndDeliver::test_fetch_yellow_cup -v

# 使用 gpt-5.4 模型运行全部 Layer 1 测试
TEST_MODEL=gpt-5.4 \
TEST_BASE_URL=https://api.chatanywhere.tech/v1 \
TEST_API_KEY=sk-OltbgRT93Ua8BGX85EgZWNV02t9WtAJt4A6xoRtAIak7IyNE \
python -m pytest examples/real_home_living/tests/ -v --ignore=examples/real_home_living/tests/test_e2e.py


# 仅运行查询类测试
python -m pytest examples/real_home_living/tests/test_task_planning.py -v -k "TestQuery"

# 仅运行整理类测试
python -m pytest examples/real_home_living/tests/test_task_planning.py -v -k "TestTidyAndFold"

# 运行跨模型对比（debug 模式）：
python run_comparison.py \
    --models "Qwen3-235B-A22B-Instruct-2507,Qwen3-VL-235B-A22B-Instruct" \
    --base-urls "http://120.48.76.60:5014/v1,http://120.48.75.178:4970/v1" \
    --include-stability \
    --debug

# 从已有 JSON 报告直接生成对比报告：
python run_comparison.py --from-reports report/eval_report_Qwen3-VL-235B-A22B-Instruct.json
# 自动发现 report/ 下所有 eval_report_*.json
python run_comparison.py --from-reports
