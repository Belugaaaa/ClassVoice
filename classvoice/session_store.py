from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
import json
import re
import uuid


SessionStatus = Literal["idle", "in_class", "paused", "ended"]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value.strip(), flags=re.UNICODE)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "class"


@dataclass
class TranscriptChunk:
    text: str
    created_at: str = field(default_factory=now_iso)
    source: str = "manual"


@dataclass
class ClassSession:
    title: str
    course_outline: str = ""
    status: SessionStatus = "idle"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    ended_at: str | None = None
    transcript: list[TranscriptChunk] = field(default_factory=list)
    notes: str = ""

    def start(self) -> None:
        self.status = "in_class"
        self.started_at = self.started_at or now_iso()

    def pause(self) -> None:
        if self.status == "in_class":
            self.status = "paused"

    def resume(self) -> None:
        if self.status == "paused":
            self.status = "in_class"

    def end(self) -> None:
        self.status = "ended"
        self.ended_at = now_iso()

    def add_transcript(self, text: str, source: str = "manual") -> None:
        clean_text = text.strip()
        if clean_text:
            self.transcript.append(TranscriptChunk(text=clean_text, source=source))

    @property
    def transcript_text(self) -> str:
        return "\n".join(chunk.text for chunk in self.transcript)


class SessionStore:
    def __init__(self, base_dir: str | Path = "data/sessions") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: ClassSession) -> tuple[Path, Path]:
        data = asdict(session)
        stem = f"{slugify(session.title)}-{session.id}"
        json_path = self.base_dir / f"{stem}.json"
        md_path = self.base_dir / f"{stem}.md"

        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.to_markdown(session), encoding="utf-8")
        return json_path, md_path

    def to_markdown(self, session: ClassSession) -> str:
        lines = [
            f"# {session.title}",
            "",
            f"- 会话 ID: `{session.id}`",
            f"- 状态: `{session.status}`",
            f"- 开始时间: {session.started_at or session.created_at}",
            f"- 结束时间: {session.ended_at or '未结束'}",
            "",
            "## 生成笔记",
            "",
            session.notes.strip() or "暂无笔记。",
            "",
            "## 课堂转写",
            "",
            session.transcript_text.strip() or "暂无转写。",
            "",
            "## 课件上下文",
            "",
            session.course_outline.strip() or "未上传课件。",
            "",
        ]
        return "\n".join(lines)
