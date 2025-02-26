import os
import sys
sys.path.append(os.getcwd())
import torch
import gradio as gr
from accelerate import Accelerator
import argparse
import spaces
from models.autoui_model import AutoUIAgent
from train_rl import DigiRLTrainer


@spaces.GPU()
def predict(text, image_path):
    image_features = image_features = torch.stack([trainer.image_process.to_feat(image_path)[..., -1408:]])

    raw_actions = trainer.agent.get_action([text], image_features.to(dtype=torch.bfloat16))
    
    return raw_actions[0]


def main(model_name):
    global trainer
    
    accelerator = Accelerator()
    device = accelerator.device

    print("### load AutoUIAgent")

    agent = AutoUIAgent(
        device=device,
        accelerator=accelerator,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=128,
        policy_lm="checkpoints/Auto-UI-Base",
        critic_lm="checkpoints/critic_1218/merge-520",
    )

    trainer = DigiRLTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=agent.tokenizer
    )

    if model_name != "autoui":
        print(f"### loading the checkpoint: {model_name}")
        trainer.load(model_name)

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.'),
            gr.Image(type="filepath", label="Image Prompt", value=None),
        ],
        outputs="text"
    )

    demo.launch(share=True, show_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()

    if args.model == "autoui":
        model_name = "checkpoints/Auto-UI-Base"
    elif args.model == "our_general":
        model_name = "checkpoints/rl-1227/epoch_13"
    elif args.model == "our_webshop":
        model_name = "checkpoints/rl-webshop/epoch_13"
    else:
        model_name = ""
    
    print(f"### model_name: {model_name}")
    main(model_name)
