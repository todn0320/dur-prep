import json
import gradio as gr

from src.pipeline.run_pipeline import run_pipeline


def predict_pill(image_path):
    if image_path is None:
        return (
            "이미지를 업로드해주세요.",
            "-",
            "-",
            "-",
            "{}"
        )

    result = run_pipeline(image_path)

    selected = result.get("drug_info", {}).get("selected_item", {})
    candidates = result.get("drug_info", {}).get("candidates", [])
    rag_text = result.get("rag_text", "")

    item_name = selected.get("item_name", "후보 없음")
    confidence = selected.get("confidence", 0.0)

    top3_lines = []
    for i, c in enumerate(candidates[:3], start=1):
        top3_lines.append(
            f"{i}. {c.get('item_name', '-')}"
            f" | item_seq={c.get('item_seq', '-')}"
            f" | final_score={c.get('final_score', '-')}"
        )
    top3_text = "\n".join(top3_lines) if top3_lines else "후보 없음"

    result_text = json.dumps(result, ensure_ascii=False, indent=2)

    return (
        item_name,
        str(confidence),
        rag_text,
        top3_text,
        result_text
    )


custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Noto Sans KR', sans-serif;
}
.block-title {
    text-align: center;
    margin-bottom: 8px;
}
.block-desc {
    text-align: center;
    color: #666;
    margin-bottom: 24px;
}
.result-card {
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    background: white;
}
"""


with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div class="block-title">
            <h1>Pill Identification PoC Demo</h1>
        </div>
        <div class="block-desc">
            알약 이미지를 업로드하면 AI 분류 → OCR → DB 조회 → 설명 생성 결과를 확인할 수 있습니다.
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="알약 이미지 업로드")
            run_button = gr.Button("예측 실행", variant="primary")

        with gr.Column(scale=1):
            final_name = gr.Textbox(label="최종 후보 약품명")
            final_conf = gr.Textbox(label="Confidence")
            rag_output = gr.Textbox(label="설명 결과", lines=8)

    with gr.Row():
        top3_output = gr.Textbox(label="Top 후보 요약", lines=6)

    with gr.Accordion("전체 파이프라인 결과(JSON)", open=False):
        json_output = gr.Textbox(lines=28, show_copy_button=True)

    run_button.click(
        fn=predict_pill,
        inputs=image_input,
        outputs=[final_name, final_conf, rag_output, top3_output, json_output],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True, show_api=False)