from __future__ import annotations

import re
from functools import lru_cache


NOTE_PROMPT = """你是一个课堂语音笔记助手。请根据课件全文和课堂转写，生成结构化中文课堂笔记。

要求：
1. 先总结本节课主题。
2. 提炼核心知识点，按小标题组织。
3. 标出老师强调的内容、例子和可能的考试/复习点。
4. 给出“待复习问题”。
5. 如果课件或转写没有明确提到，不要编造事实。

课件全文：
{course_text}

课堂转写：
{transcript}

请输出课堂笔记：
"""


def simple_note_generator(course_text: str, transcript: str) -> str:
    course_preview = _first_lines(course_text, limit=8)
    transcript_preview = _first_lines(transcript, limit=16)
    keywords = _pick_keywords(course_text + "\n" + transcript)

    sections = [
        "## 本节课概览",
        "- 当前使用轻量规则模式生成笔记。",
        f"- 可能主题关键词：{', '.join(keywords) if keywords else '暂无足够文本'}",
        "",
        "## 课件内容线索",
        course_preview or "- 未上传或未解析到课件文本。",
        "",
        "## 课堂内容摘录",
        transcript_preview or "- 暂无课堂转写。",
        "",
        "## 待复习问题",
        "- 结合课件检查关键概念定义是否完整。",
        "- 回看课堂转写中出现频率较高的术语。",
    ]
    return "\n".join(sections)


def generate_notes(
    outline: str,
    transcript: str,
    mode: str = "simple",
    model_name: str = "Qwen/Qwen3-0.6B",
    max_new_tokens: int = 700,
    max_input_chars: int = 16000,
    enable_thinking: bool = False,
    temperature: float = 0.7,
) -> str:
    if not transcript.strip() and not outline.strip():
        return "暂无可用于生成笔记的课件或转写内容。"

    if mode in {"qwen-local", "transformers"}:
        return generate_notes_with_qwen(
            course_text=outline,
            transcript=transcript,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            max_input_chars=max_input_chars,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )
    return simple_note_generator(outline, transcript)


def generate_notes_with_qwen(
    course_text: str,
    transcript: str,
    model_name: str,
    max_new_tokens: int,
    max_input_chars: int,
    enable_thinking: bool,
    temperature: float,
) -> str:
    try:
        tokenizer, model = _load_qwen_model(model_name)
    except Exception as exc:
        return (
            "本地 Qwen 模型加载失败，已回退到轻量规则模式。\n\n"
            f"错误信息：`{exc}`\n\n"
            + simple_note_generator(course_text, transcript)
        )

    prepared_course_text = _fit_text(course_text, max_input_chars)
    prepared_transcript = _fit_text(transcript, max_input_chars // 2)
    prompt = NOTE_PROMPT.format(
        course_text=prepared_course_text or "未提供课件。",
        transcript=prepared_transcript or "暂无转写。",
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.8 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _strip_thinking(output) or "模型没有生成有效内容，请尝试减少输入长度或增大生成长度。"


@lru_cache(maxsize=2)
def _load_qwen_model(model_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "请先安装 Qwen 依赖：pip install transformers>=4.52.4 torch accelerate"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return tokenizer, model


def check_qwen_ready(model_name: str) -> str:
    try:
        tokenizer, model = _load_qwen_model(model_name)
    except Exception as exc:
        return f"Qwen 未就绪：{exc}"
    return f"Qwen 已就绪：{model_name}，device={model.device}，vocab={len(tokenizer)}"


def _fit_text(text: str, max_chars: int) -> str:
    clean = text.strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean

    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return (
        clean[:head_chars]
        + "\n\n...[中间课件文本过长，已为本次模型输入省略；原文仍保存在课堂记录中]...\n\n"
        + clean[-tail_chars:]
    )


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _first_lines(text: str, limit: int) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(f"- {line[:180]}" for line in lines[:limit])


def _pick_keywords(text: str) -> list[str]:
    tokens: dict[str, int] = {}
    for raw in text.replace("\n", " ").split():
        token = raw.strip("，。！？；：,.!?;:()[]【】")
        if len(token) < 2 or token.isdigit():
            continue
        tokens[token] = tokens.get(token, 0) + 1
    return [word for word, _ in sorted(tokens.items(), key=lambda item: item[1], reverse=True)[:8]]
