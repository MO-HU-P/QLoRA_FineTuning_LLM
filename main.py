
import gradio as gr
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained(
    "qlora_model",
    torch_dtype=torch.float16,
    load_in_4bit=True, 
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct",  
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"


def generate_response(message, history, system_prompt, max_tokens, temperature):

    prompt = f"[INST] <<SYS>>\n {system_prompt} \n<</SYS>>\n {message} [/INST]"

    input_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, return_tensors='pt').input_ids.cuda()
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
    )
    output = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)

    return output

iface = gr.ChatInterface(fn=generate_response, 
                     additional_inputs=[
                         gr.Textbox("日本語で回答してください。", label="system_prompt"), 
                         gr.Slider(50, 500, 250, label="max_tokens"),
                         gr.Slider(0, 1.0, 0.3, label="temperature"),
                     ], 
                     title="Chat with QLoRA model",
                     retry_btn=None,
                     undo_btn="Delete Previous",
                     clear_btn="Clear",
                    )

iface.launch()