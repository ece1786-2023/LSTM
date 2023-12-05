# coding:utf-8
'''
**************************************************
@File   ：LSTM -> demo
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , Gradio demo
@Date   ：2023/12/4 10:50
**************************************************
'''

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def story_generator(title, name, gender, attr1, attr1_num, attr2, attr2_num, attr3, attr3_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/ft1"
    attr = attr1 + '+' + str(attr1_num) + '\t' + attr2 + '+' + str(attr2_num) + '\t' + attr3 + '+' + str(attr3_num)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
    prompt = "This is the story of [PAWN_nameDef], a " + title + " with " + attr + ": "
    trun = len(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, do_sample=True, max_length=100, temperature=1, top_p=0.5,
                             repetition_penalty=1)
    generate_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generate_sentence = generate_sentence[0][trun:].strip('"')
    generate_sentence = generate_sentence.replace('[PAWN_nameDef]', name)
    if '[PAWN_pronoun]' in generate_sentence:
        if gender == 'Male':
            generate_sentence = generate_sentence.replace('[PAWN_pronoun]', 'He')
        else:
            generate_sentence = generate_sentence.replace('[PAWN_pronoun]', 'She')
    if '[PAWN_possessive]' in generate_sentence:
        if gender == 'Male':
            generate_sentence = generate_sentence.replace('[PAWN_possessive]', 'his')
        else:
            generate_sentence = generate_sentence.replace('[PAWN_possessive]', 'her')
    if '[PAWN_objective]' in generate_sentence:
        if gender == 'Male':
            generate_sentence = generate_sentence.replace('[PAWN_objective]', 'him')
        else:
            generate_sentence = generate_sentence.replace('[PAWN_objective]', 'her')
    return generate_sentence

if __name__ == '__main__':
    attr_ls = ['Cooking', 'Driving', 'Shooting', 'Social', 'Artistic', 'Medicine', 'Intellectual', 'Crafting', 'Plants',
               'Construction']
    title = gr.Text(label="Character Title")
    name = gr.Text(label="Character Name")
    gender = gr.Radio(choices=['Male', 'Female'], label="Character Gender")
    attr1 = gr.Dropdown(choices=attr_ls, label="Select One Attribute", info="Select an Attribute for the Character")
    attr1_num = gr.Number(label="Attribute 1 Value", info="Please enter the value here", minimum=0, maximum=10)
    attr2 = gr.Dropdown(choices=attr_ls, label="Select One Attribute", info="Select an Attribute for the Character")
    attr2_num = gr.Number(label="Attribute 2 Value", info="Please enter the value here", minimum=0, maximum=10)
    attr3 = gr.Dropdown(choices=attr_ls, label="Select One Attribute", info="Select an Attribute for the Character")
    attr3_num = gr.Number(label="Attribute 3 Value", info="Please enter the value here", minimum=0, maximum=10)
    output = gr.Text(label="Background Description")
    demo = gr.Interface(fn=story_generator,
                        inputs=[title, name, gender, attr1, attr1_num, attr2, attr2_num, attr3, attr3_num],
                        outputs=output)
    demo.launch()
