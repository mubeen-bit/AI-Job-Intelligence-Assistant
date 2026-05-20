import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(context):

    html = """
    <div style="padding:10px;">
        <h2>Retrieved Context</h2>
    """

    for i, doc in enumerate(context, start=1):

        source = doc.metadata.get("source", "Unknown")

        html += f"""
        <div style="
            border:1px solid #ccc;
            border-radius:10px;
            padding:10px;
            margin-bottom:15px;
        ">

        <h4>Context {i}</h4>

        <p>
            <strong>Source:</strong> {source}
        </p>

        <div style="
            white-space: pre-wrap;
            margin-top:10px;
        ">
            {doc.page_content}
        </div>

        </div>
        """

    html += "</div>"

    return html


def put_message_in_chatbot(message, history):

    history = history or []

    history.append([message, None])

    return "", history


def chat(history):

    user_message = history[-1][0]

    answer, context = answer_question(user_message)

    history[-1][1] = answer

    return history, format_context(context)


def main():

    with gr.Blocks(title="AI Job Intelligence Assistant") as ui:

        gr.Markdown("""
# AI Job Intelligence Assistant

Ask questions about:
- job descriptions
- required skills
- technologies
- hiring trends
- career paths

### Example Questions

- What skills are required for Data Analyst roles?
- Compare Django vs Flask jobs
- What tools are common in ML Engineer positions?
""")

        with gr.Row():

            with gr.Column(scale=1):

                chatbot = gr.Chatbot(
                    label="Career Assistant",
                    height=600,
                )

                message = gr.Textbox(
                    placeholder="Ask about jobs, frameworks, skills, or careers...",
                    show_label=False,
                )

            with gr.Column(scale=1):

                context_panel = gr.HTML(
                    value="""
                    <div style="padding:10px;">
                        Retrieved context will appear here.
                    </div>
                    """
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_panel],
        )

    ui.launch(share=True)


if __name__ == "__main__":
    main()