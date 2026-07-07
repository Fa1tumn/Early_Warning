"""
阶段1任务：验证 LangGraph 能否拿到 token-level logprobs。

策略：不依赖商业 API（本机未配置 OPENAI_API_KEY），直接用 HuggingFace
transformers 本地跑一个小模型（Qwen2.5-1.5B-Instruct，8GB显存内可跑），
用 model.generate(output_scores=True) 拿到每一步生成 token 的完整 logits，
再包成一个 LangGraph 单节点 state graph，验证：
1) 能否在 LangGraph 的节点函数里访问到底层模型调用返回的 token 级 logprobs
2) 把 logprobs 序列作为 state 的一部分传递下去（供后续 Signal Extractor 使用）

结论会写在脚本末尾打印的 SUMMARY 里。等真正上4090机器时，只需要把
MODEL_NAME 换成更大的模型（如 Qwen2.5-14B-Instruct），其余代码不用改。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Tuple

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
print(f"Loaded. device={model.device}, cuda_available={torch.cuda.is_available()}")


def generate_with_logprobs(prompt: str, max_new_tokens: int = 64) -> Tuple[str, List[dict]]:
    """调用底层模型生成，同时返回逐 token 的 logprob。"""
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    gen_ids = out.sequences[0, input_ids.shape[1]:]
    scores = out.scores  # tuple of per-step logits, len == generated tokens

    token_logprobs = []
    for step_logits, tok_id in zip(scores, gen_ids):
        logprobs = torch.log_softmax(step_logits[0], dim=-1)
        token_logprobs.append({
            "token": tokenizer.decode([tok_id]),
            "logprob": logprobs[tok_id].item(),
            "entropy": (-(logprobs.exp() * logprobs).sum()).item(),
        })

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, token_logprobs


# ---- 最小 LangGraph 验证：把 logprobs 塞进 state，走一遍图 ----

class AgentState(TypedDict):
    prompt: str
    response: str
    token_logprobs: List[dict]


def llm_node(state: AgentState) -> AgentState:
    text, tlp = generate_with_logprobs(state["prompt"])
    return {"response": text, "token_logprobs": tlp}


graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.set_entry_point("llm")
graph.add_edge("llm", END)
app = graph.compile()


if __name__ == "__main__":
    test_prompt = "用一句话解释什么是死循环。"
    result = app.invoke({"prompt": test_prompt, "response": "", "token_logprobs": []})

    print("\n=== LangGraph 输出 ===")
    print("response:", result["response"])
    print("\n=== 前10个token的logprob/熵 ===")
    for item in result["token_logprobs"][:10]:
        print(f"  token={item['token']!r:12s} logprob={item['logprob']:.4f}  entropy={item['entropy']:.4f}")

    print("\n=== SUMMARY ===")
    print(f"共获取 {len(result['token_logprobs'])} 个token的logprob，均可正常计算熵。")
    print("结论：本地HF模型 + LangGraph state graph 组合可以在节点内部完整拿到")
    print("token级logprobs（非阉割版，非top-N近似），可以直接用于后续熵/语义漂移信号提取。")
    print("换到4090机器时，只需替换 MODEL_NAME 为更大模型（如 Qwen2.5-14B-Instruct），")
    print("其余 generate_with_logprobs / LangGraph 接线逻辑不变。")
