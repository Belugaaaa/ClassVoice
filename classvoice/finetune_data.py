from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
import json
import uuid


FINETUNE_DIR = Path("data/finetune")
UPLOAD_DIR = FINETUNE_DIR / "uploads"
DATASET_PATH = FINETUNE_DIR / "qlora_notes.jsonl"


@dataclass
class FineTuneSample:
    id: str
    created_at: str
    title: str
    transcript: str
    course_text: str
    note: str
    source_files: list[str] = field(default_factory=list)

    def to_messages(self) -> list[dict[str, str]]:
        user_parts = [
            "请根据以下网课材料生成结构化课堂笔记。",
            "",
            f"课程标题：{self.title or '未命名网课'}",
        ]
        if self.course_text.strip():
            user_parts.extend(["", "课件/文本材料：", self.course_text.strip()])
        if self.transcript.strip():
            user_parts.extend(["", "视频/音频转写：", self.transcript.strip()])
        if self.source_files:
            user_parts.extend(["", "原始媒体文件：", "\n".join(f"- {item}" for item in self.source_files)])

        return [
            {
                "role": "system",
                "content": "你是一个课堂语音笔记助手，擅长把网课材料整理成结构化中文课堂笔记。",
            },
            {"role": "user", "content": "\n".join(user_parts).strip()},
            {"role": "assistant", "content": self.note.strip()},
        ]

    def to_jsonl_record(self) -> dict:
        payload = asdict(self)
        payload["messages"] = self.to_messages()
        return payload


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_uploaded_file(file: BinaryIO, sample_id: str) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = getattr(file, "name", "uploaded_file")
    safe_name = Path(name).name
    target = UPLOAD_DIR / sample_id / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)

    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    target.write_bytes(data)
    return target.as_posix()


def append_finetune_sample(
    title: str,
    transcript: str,
    course_text: str,
    note: str,
    uploaded_files: list[BinaryIO] | None = None,
) -> FineTuneSample:
    sample_id = uuid.uuid4().hex[:10]
    source_files = []
    for file in uploaded_files or []:
        source_files.append(save_uploaded_file(file, sample_id))

    sample = FineTuneSample(
        id=sample_id,
        created_at=now_iso(),
        title=title.strip() or "未命名网课",
        transcript=transcript.strip(),
        course_text=course_text.strip(),
        note=note.strip(),
        source_files=source_files,
    )
    append_jsonl(sample)
    return sample


def append_jsonl(sample: FineTuneSample, dataset_path: Path = DATASET_PATH) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(sample.to_jsonl_record(), ensure_ascii=False) + "\n")


def count_samples(dataset_path: Path = DATASET_PATH) -> int:
    if not dataset_path.exists():
        return 0
    with dataset_path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def dataset_preview(dataset_path: Path = DATASET_PATH, limit: int = 3) -> list[dict]:
    if not dataset_path.exists():
        return []

    records = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records
