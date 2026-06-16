from __future__ import annotations

import streamlit as st

from classvoice.llm import check_qwen_ready, generate_notes
from classvoice.pdf_utils import extract_pdf_text
from classvoice.finetune_data import DATASET_PATH, append_finetune_sample, count_samples, dataset_preview
from classvoice.session_store import ClassSession, SessionStore
from classvoice.speech import (
    SpeechConfig,
    VoskRecorder,
    get_default_input_device,
    list_input_devices,
    measure_microphone_level,
)


st.set_page_config(page_title="ClassVoice", page_icon="CV", layout="wide")


def get_session() -> ClassSession:
    if "class_session" not in st.session_state:
        st.session_state.class_session = ClassSession(title="未命名课程")
    return st.session_state.class_session


def set_session(session: ClassSession) -> None:
    st.session_state.class_session = session


def get_recorder(config: SpeechConfig) -> VoskRecorder:
    old_config = st.session_state.get("vosk_config")
    recorder = st.session_state.get("vosk_recorder")
    if recorder is None or old_config != config:
        if recorder is not None and recorder.is_running:
            recorder.stop()
        recorder = VoskRecorder(config)
        st.session_state.vosk_recorder = recorder
        st.session_state.vosk_config = config
    return recorder


def stop_recorder_if_running() -> None:
    recorder = st.session_state.get("vosk_recorder")
    if recorder is not None and recorder.is_running:
        recorder.stop()


def pull_vosk_results(recorder: VoskRecorder, session: ClassSession) -> None:
    statuses = recorder.drain_statuses()
    if statuses:
        st.session_state.last_vosk_status = statuses[-1]

    for text in recorder.drain_texts():
        session.add_transcript(text, source="vosk")

    partials = recorder.drain_partials()
    if partials:
        st.session_state.last_vosk_partial = partials[-1]

    for error in recorder.drain_errors():
        st.warning(f"Vosk 识别提示：{error}")


@st.fragment(run_every="1s")
def render_transcript_panel(
    recorder: VoskRecorder,
    session: ClassSession,
    auto_refresh_vosk: bool,
) -> None:
    if recorder.is_running and auto_refresh_vosk:
        pull_vosk_results(recorder, session)

    if recorder.is_running:
        st.caption("录音中。已开启自动刷新时，识别结果会大约每秒更新一次。")
    if "last_vosk_status" in st.session_state:
        st.caption(f"Vosk 状态：{st.session_state.last_vosk_status}")
    if "last_vosk_partial" in st.session_state:
        st.info(f"最近临时识别：{st.session_state.last_vosk_partial}")

    st.markdown("#### 已记录转写")
    if session.transcript:
        for chunk in reversed(session.transcript[-8:]):
            st.write(f"**{chunk.created_at}** · {chunk.source}")
            st.write(chunk.text)
    else:
        st.caption("暂无转写。点击上课后可以添加课堂内容，或启动 Vosk 录音。")


store = SessionStore()
session = get_session()

st.title("ClassVoice")
st.caption("本地课堂语音笔记助手")

with st.sidebar:
    st.header("课堂设置")
    title = st.text_input("课程/课堂标题", value=session.title)
    if title != session.title and session.status in {"idle", "ended"}:
        session.title = title

    uploaded_pdf = st.file_uploader("上传课件 PDF", type=["pdf"])
    if uploaded_pdf is not None and st.button("解析课件", use_container_width=True):
        with st.spinner("正在解析 PDF..."):
            pdf_result = extract_pdf_text(uploaded_pdf.getvalue())
            session.course_outline = pdf_result.text
        st.success(
            f"课件全文解析完成：共 {pdf_result.page_count} 页，"
            f"抽取到 {pdf_result.extracted_pages} 页文本，约 {pdf_result.char_count} 字符。"
        )

    st.divider()
    st.header("笔记生成")
    llm_mode = st.selectbox("LLM 模式", ["simple", "qwen-local"], index=1)
    model_name = st.text_input("模型名/本地路径", value="models/qwen3-0.6b")
    max_new_tokens = st.slider("最大生成长度", min_value=128, max_value=1600, value=700, step=64)
    max_input_chars = st.slider("送入模型的课件字符上限", min_value=4000, max_value=32000, value=16000, step=1000)
    enable_thinking = st.checkbox("启用 Qwen thinking", value=False)
    temperature = st.slider("生成温度", min_value=0.0, max_value=1.2, value=0.7, step=0.1)
    if st.button("检查 Qwen 状态", use_container_width=True):
        with st.spinner("正在加载/检查 Qwen 模型..."):
            st.session_state.qwen_status = check_qwen_ready(model_name)
    if "qwen_status" in st.session_state:
        st.info(st.session_state.qwen_status)

    st.divider()
    st.header("Vosk 语音识别")
    vosk_model_path = st.text_input("Vosk 模型目录", value="models/vosk-model-small-cn-0.22")
    auto_refresh_vosk = st.checkbox("录音时自动刷新识别结果", value=True)

    input_devices = list_input_devices()
    default_input = get_default_input_device()
    device_options = ["默认麦克风"]
    device_map: dict[str, int | None] = {"默认麦克风": None}
    for device in input_devices:
        label = (
            f"{device['index']}: {device['name']} "
            f"({device['hostapi']}, {device['channels']}ch, {device['default_samplerate']}Hz)"
        )
        device_options.append(label)
        device_map[label] = int(device["index"])

    selected_device = st.selectbox("输入设备", device_options, index=0)
    device_id = device_map[selected_device]

    if default_input is not None:
        st.caption(f"系统默认输入设备编号：{default_input}")

    selected_info = next((item for item in input_devices if item["index"] == device_id), None)
    default_rate = selected_info["default_samplerate"] if selected_info else 16000
    sample_rate = st.number_input(
        "采样率",
        min_value=8000,
        max_value=48000,
        value=int(default_rate or 16000),
        step=1000,
        help="如果录音失败，优先尝试设备标签里显示的默认采样率。",
    )

    vosk_config = SpeechConfig(
        model_path=vosk_model_path,
        sample_rate=int(sample_rate),
        device=device_id,
    )
    recorder = get_recorder(vosk_config)

    if st.button("测试麦克风音量", use_container_width=True):
        with st.spinner("请对着麦克风说话，正在录制 2 秒..."):
            try:
                level = measure_microphone_level(device_id, int(sample_rate), seconds=2.0)
                st.session_state.mic_level = level
            except Exception as exc:
                st.session_state.mic_level_error = str(exc)

    if "mic_level" in st.session_state:
        level = st.session_state.mic_level
        st.write(f"RMS: `{level['rms']:.5f}`")
        st.write(f"Peak: `{level['peak']:.5f}`")
        if level["peak"] < 0.005:
            st.warning("几乎没有收到声音，建议换输入设备或检查系统麦克风权限。")
        else:
            st.success("麦克风有输入信号。")
    if "mic_level_error" in st.session_state:
        st.error(st.session_state.mic_level_error)


pull_vosk_results(recorder, session)

status_label = {
    "idle": "未开始",
    "in_class": "上课中",
    "paused": "课间暂停",
    "ended": "已下课",
}

top_cols = st.columns([1.2, 1, 1, 1, 1])
top_cols[0].metric("当前状态", status_label[session.status])

if top_cols[1].button("上课", disabled=session.status == "in_class", use_container_width=True):
    if session.status == "ended":
        stop_recorder_if_running()
        session = ClassSession(title=title or "未命名课程", course_outline=session.course_outline)
        set_session(session)
    session.title = title or session.title
    session.start()
    st.rerun()

if top_cols[2].button("课间暂停", disabled=session.status != "in_class", use_container_width=True):
    stop_recorder_if_running()
    session.pause()
    st.rerun()

if top_cols[3].button("继续上课", disabled=session.status != "paused", use_container_width=True):
    session.resume()
    st.rerun()

if top_cols[4].button("下课并保存", disabled=session.status not in {"paused", "in_class"}, use_container_width=True):
    stop_recorder_if_running()
    session.end()
    json_path, md_path = store.save(session)
    st.toast(f"已保存：{md_path}")
    st.session_state.last_saved = (str(json_path), str(md_path))
    st.rerun()

if "last_saved" in st.session_state:
    json_path, md_path = st.session_state.last_saved
    st.success(f"最近保存：{md_path}")

main_left, main_right = st.columns([1.05, 0.95])

with main_left:
    st.subheader("课堂转写")

    speech_cols = st.columns([1, 1, 1, 1.2])
    can_record = session.status == "in_class"
    if speech_cols[0].button("开始 Vosk 录音", disabled=not can_record or recorder.is_running, use_container_width=True):
        recorder.start()
        st.rerun()
    if speech_cols[1].button("停止 Vosk 录音", disabled=not recorder.is_running, use_container_width=True):
        recorder.stop()
        pull_vosk_results(recorder, session)
        st.rerun()
    if speech_cols[2].button("拉取识别结果", disabled=not recorder.is_running, use_container_width=True):
        pull_vosk_results(recorder, session)
        st.rerun()
    speech_cols[3].metric("录音状态", "识别中" if recorder.is_running else "未录音")

    transcript_input = st.text_area(
        "输入或粘贴课堂语音识别文本",
        height=180,
        placeholder="例如：今天我们讲语音交互系统的基本流程，包括唤醒、识别、理解和反馈...",
        disabled=session.status not in {"in_class", "paused"},
    )
    add_disabled = session.status not in {"in_class", "paused"} or not transcript_input.strip()
    if st.button("添加到本节课转写", disabled=add_disabled, use_container_width=True):
        session.add_transcript(transcript_input)
        st.rerun()

    render_transcript_panel(recorder, session, auto_refresh_vosk)

with main_right:
    st.subheader("课堂笔记")
    if st.button("生成/刷新笔记", use_container_width=True):
        if not session.course_outline.strip() and not session.transcript_text.strip():
            st.warning("还没有课件全文或课堂转写，无法生成笔记。请先上传 PDF、录音识别，或手动添加转写。")
        else:
            with st.spinner("正在生成笔记..."):
                session.notes = generate_notes(
                    outline=session.course_outline,
                    transcript=session.transcript_text,
                    mode=llm_mode,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    max_input_chars=max_input_chars,
                    enable_thinking=enable_thinking,
                    temperature=temperature,
                )
    st.markdown(session.notes or "暂无笔记。")

st.divider()
bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.subheader("课件全文")
    st.text_area(
        "PDF 全文解析文本",
        value=session.course_outline,
        height=220,
        disabled=False,
        key="outline_view",
    )
    if st.button("使用编辑后的课件全文", use_container_width=True):
        session.course_outline = st.session_state.outline_view
        st.success("已更新课件全文")

with bottom_right:
    st.subheader("本地保存")
    st.write("下课时会自动保存 JSON 和 Markdown。也可以随时手动保存当前进度。")
    if st.button("保存当前进度", use_container_width=True):
        json_path, md_path = store.save(session)
        st.session_state.last_saved = (str(json_path), str(md_path))
        st.success(f"已保存：{md_path}")

    st.code("data/sessions/", language="text")

st.divider()
st.subheader("QLoRA 微调数据接口")
st.write("上传网课视频、音频、文本材料和对应人工笔记，生成本地 JSONL 训练集。训练集和媒体文件只保存在本地，不提交 Git。")

ft_left, ft_right = st.columns([1, 1])

with ft_left:
    ft_title = st.text_input("样本标题", placeholder="例如：语音交互导论 第 1 讲")
    ft_files = st.file_uploader(
        "上传网课视频/音频/文本",
        type=["mp4", "mov", "mkv", "mp3", "wav", "m4a", "aac", "txt", "md"],
        accept_multiple_files=True,
    )
    ft_course_text = st.text_area(
        "课程文本/课件文本",
        height=140,
        placeholder="可以粘贴字幕、课件文本、视频简介等。",
    )
    ft_transcript = st.text_area(
        "视频/音频转写",
        height=160,
        placeholder="可以粘贴人工转写，或后续由 Vosk/其他 ASR 生成的转写。",
    )

with ft_right:
    ft_note = st.text_area(
        "对应标准笔记",
        height=260,
        placeholder="这里填写你希望模型学习输出的高质量课堂笔记。",
    )
    add_ft_disabled = not ft_note.strip() or (not ft_course_text.strip() and not ft_transcript.strip() and not ft_files)
    if st.button("追加为 QLoRA 训练样本", disabled=add_ft_disabled, use_container_width=True):
        sample = append_finetune_sample(
            title=ft_title,
            transcript=ft_transcript,
            course_text=ft_course_text,
            note=ft_note,
            uploaded_files=ft_files,
        )
        st.success(f"已添加训练样本：{sample.id}")

    st.metric("当前训练样本数", count_samples())
    st.code(str(DATASET_PATH), language="text")

    st.markdown("#### 训练命令")
    st.code(
        "pip install -r requirements-finetune.txt\n"
        "python scripts/train_qlora.py --model-path models/qwen3-0.6b --dataset data/finetune/qlora_notes.jsonl",
        language="powershell",
    )

preview_records = dataset_preview()
if preview_records:
    with st.expander("查看训练集预览"):
        for record in preview_records:
            st.json(
                {
                    "id": record.get("id"),
                    "title": record.get("title"),
                    "source_files": record.get("source_files", []),
                    "messages": record.get("messages", [])[:3],
                }
            )
