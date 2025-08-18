# Minimal test to isolate the Gradio issue
import gradio as gr

def dummy_process(video_path):
    if video_path is None:
        return None, "No video uploaded"
    return video_path, f"Received video: {video_path}"

# Test with minimal interface
with gr.Blocks() as demo:
    gr.Markdown("## Test Interface")
    
    with gr.Row():
        inp = gr.Video(label="Upload video")
        out_vid = gr.Video(label="Output")
        out_txt = gr.Textbox(label="Status")
    
    btn = gr.Button("Test")
    btn.click(dummy_process, inputs=inp, outputs=[out_vid, out_txt])

if __name__ == "__main__":
    demo.launch()