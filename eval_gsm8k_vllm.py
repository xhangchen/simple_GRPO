"""
用 vLLM 对 GSM8K 测试集做快速评测（accuracy + format 合规率）。

用法示例：
  CUDA_VISIBLE_DEVICES=0 python eval_gsm8k_vllm.py --model ./step_1000
  CUDA_VISIBLE_DEVICES=0 python eval_gsm8k_vllm.py --model ./step_1000 --limit 100
  CUDA_VISIBLE_DEVICES=0 python eval_gsm8k_vllm.py --model ./step_1000 --limit 100 --out_jsonl ./gsm8k_pred.jsonl

说明：
- `--model` 指向训练保存的断点目录（HuggingFace save_pretrained 格式）。
- accuracy：从模型输出中提取最终数字答案，与 GSM8K 的 ground truth 做等价验证。
- format_rate：输出是否符合 `<think>...</think><answer>...</answer>` 结构及标签数量约束。
- `--out_jsonl`：可选，把每条样本的生成结果/解析结果落盘，便于排查 format_rate=0 的原因。
"""

import json
import re
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from math_verify import parse, verify, ExprExtractionConfig

# 与训练脚本保持一致的 system prompt（用于 chat template 拼接）
SYSTEM_PROMPT = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

# format 合规：必须以 <think>... </think> + <answer>... </answer> 结尾（中间允许换行/空格）
FORMAT_PATTERN = re.compile(r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$", re.DOTALL | re.VERBOSE)
# 数字抽取：兼容小数 / 分数 / 整数，评测时取最后一个匹配到的数字
NUM_PATTERN = re.compile(r"\d+\.\d+|\d+/\d+|\d+")

def format_ok(text: str) -> bool:
    """判定输出是否满足训练时的格式要求（结构 + 标签数量）。"""
    t = text.strip()
    if not FORMAT_PATTERN.match(t):
        return False
    think_count = t.count("<think>") + t.count("</think>")
    answer_count = t.count("<answer>") + t.count("</answer>")
    return think_count == 2 and answer_count == 2

def extract_for_math(text: str) -> str | None:
    """
    从模型输出中提取用于数学验证的“最终答案字符串”。

    逻辑：
    - 优先从 <answer>...</answer> 内部抽取；
    - 否则在全文里找数字；
    - 返回最后一个匹配到的数字（与训练脚本 reward_correct 的策略一致）。
    """
    t = text
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        t = m.group(1)
    nums = NUM_PATTERN.findall(t)
    return nums[-1] if nums else None

def main():
    ap = argparse.ArgumentParser()
    # HuggingFace 模型目录（例如 ./step_1000）
    ap.add_argument("--model", default="./step_1000")
    # 生成长度上限：越大越慢，也更容易生成冗长思维链；与训练时 max_tokens 保持一致更可比
    ap.add_argument("--max_tokens", type=int, default=700)
    # 只评测前 N 条样本，便于先快速 sanity check；0 表示全量 test 集
    ap.add_argument("--limit", type=int, default=0, help="0 means full test set")
    # 把每条样本的生成结果写入 jsonl（包含 question/gt/pred/raw_text/format_ok/is_correct）
    ap.add_argument("--out_jsonl", default="", help="optional: path to write predictions as jsonl")
    # 是否把完整 prompt 一起写进 jsonl（方便复现，但文件会很大）
    ap.add_argument("--save_prompt", action="store_true", help="if set, also save full prompt into jsonl")
    # 打印前 N 条 format 不合规的样本（快速肉眼排查）
    ap.add_argument("--print_bad", type=int, default=0, help="print first N non-format outputs (0 disables)")
    args = ap.parse_args()

    # tokenizer 用来 apply_chat_template；trust_remote_code=True 便于 Qwen 等自定义实现正常加载
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # vLLM 直接加载断点目录推理
    llm = LLM(model=args.model, trust_remote_code=True)

    # GSM8K 官方 test split（1319 条）
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = []
    gts = []
    questions = []
    for ex in ds:
        q = ex["question"]
        # GSM8K answer 格式类似："... #### 42"，我们只取 #### 后面的最终答案
        gt = ex["answer"].split("####")[-1].strip()
        # 构造与训练时一致的 chat prompt
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        gts.append(gt)
        questions.append(q)

    # 确定性解码（temperature=0），更适合评测；max_tokens 控制输出长度
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    correct = 0
    fmt = 0
    total = len(prompts)

    # 批量生成（vLLM 内部会自动做 batching）；use_tqdm=True 显示进度条
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    fout = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    bad_printed = 0
    for idx, (out, gt) in enumerate(zip(outputs, gts)):
        text = out.outputs[0].text
        # 统计 format 合规率
        is_format_ok = format_ok(text)
        if is_format_ok:
            fmt += 1
        elif args.print_bad and bad_printed < args.print_bad:
            print(f"\n[bad_format #{bad_printed + 1}] idx={idx}")
            print("question:", questions[idx])
            print("raw_text:", text)
            bad_printed += 1

        # 抽取最终答案（数字）；抽不到则按错误处理
        pred = extract_for_math(text)
        is_correct = False

        try:
            if pred is not None:
                # 用 math_verify 做“等价验证”（能处理 1/2 vs 0.5 这类等价表达）
                ans = parse(pred, extraction_config=[ExprExtractionConfig()])
                ground_truth = parse(gt, extraction_config=[ExprExtractionConfig()])
                is_correct = bool(verify(ans, ground_truth))
        except Exception:
            # parse/verify 失败（格式异常等）按错误计
            pass
        if is_correct:
            correct += 1

        if fout is not None:
            rec = {
                "idx": idx,
                "question": questions[idx],
                "gt": gt,
                "pred_extracted": pred,
                "is_correct": is_correct,
                "format_ok": is_format_ok,
                "raw_text": text,
            }
            if args.save_prompt:
                rec["prompt"] = prompts[idx]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if fout is not None:
        fout.close()

    print(f"model={args.model}")
    print(f"total={total}")
    print(f"accuracy={correct/total:.4f} ({correct}/{total})")
    print(f"format_rate={fmt/total:.4f} ({fmt}/{total})")
    if args.out_jsonl:
        print(f"saved_jsonl={args.out_jsonl}")

if __name__ == "__main__":
    main()
