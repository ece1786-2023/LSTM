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


def story_generator(title, name, gender, replace, attr1, attr1_num, attr2, attr2_num, attr3, attr3_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/ft1"
    attr = attr1 + '+' + str(int(attr1_num)) + ', ' + attr2 + '+' + str(int(attr2_num)) + ', ' + attr3 + '+' + str(int(attr3_num))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
    input_sentence = "This is the story of [PAWN_nameDef], a " + title + " with " + attr.replace("+-","-") + ": "
    print(input_sentence)
    trun = len(input_sentence)
    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids.to(device)
    length_check=0 #This is the length of the generated sequence in bytes.
    while length_check<180:
        outputs = model.generate(input_ids, do_sample=True, max_length=200, temperature=1, top_p=1)
        generate_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generate_sentence = generate_sentence[0][trun:].replace('"','').replace('\xa0'," ")
        length_check=len(generate_sentence)
        print("length :", length_check, "\noutput: \n", generate_sentence)
        if replace == 'Yes':
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
    attr_ls = ['artistic', 'animals', 'construction', 'cooking', 'crafting', 'intellectual', 'medicine', 'melee',
               'mining', 'plants', 'shooting', 'social']
    title = gr.Text(label="Character Title")
    name = gr.Text(label="Character Name")
    gender = gr.Radio(choices=['Male', 'Female'], label="Character Gender")
    replace = gr.Radio(choices=['Yes', 'No'], label="Peplace Special Tokens?")
    attr1 = gr.Dropdown(choices=attr_ls, label="Select a skill modifier", info="Select a skill modifier for the character")
    attr1_num = gr.Number(label="Skill modifier 1 Value", info="Please enter the value here", minimum=-9, maximum=9)
    attr2 = gr.Dropdown(choices=attr_ls, label="Select a skill modifier", info="Select a skill modifier for the character")
    attr2_num = gr.Number(label="Skill modifier 2 Value", info="Please enter the value here", minimum=-9, maximum=9)
    attr3 = gr.Dropdown(choices=attr_ls, label="Select a skill modifier", info="Select a skill modifier for the character")
    attr3_num = gr.Number(label="Skill modifier 3 Value", info="Please enter the value here", minimum=-9, maximum=9)
    output = gr.Text(label="Background Description")
    demo = gr.Interface(fn=story_generator,
                        inputs=[title, name, gender, replace, attr1, attr1_num, attr2, attr2_num, attr3, attr3_num],
                        outputs=output)
    demo.launch(inbrowser=True)
