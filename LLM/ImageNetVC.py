import os
import copy
import json
import pandas
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, OPTForCausalLM

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def write_json(results, subset='color', model_name="llama-7B", path='./results', prompt_idx=0):
    model_path = os.path.join(path, model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    subset_path = os.path.join(model_path, subset)
    if not os.path.exists(subset_path):
        os.mkdir(subset_path)
    save_path = os.path.join(subset_path, "prompt{}".format(prompt_idx) + '.json')
    with open(save_path, "w") as f:
        json.dump(results, f)


def load_candidates(subset='color'):
    if subset == 'color':
        candidates = ['brown', 'black', 'white', 'yellow', 'green', 'gray', 'red', 'orange', 'blue', 'silver', 'pink']
    elif subset == 'shape':
        candidates = ['round', 'rectangle', 'triangle', 'square', 'oval', 'curved', 'cylinder', 'straight',
                      'cone', 'curly', 'heart', 'star']
    elif subset == 'material':
        candidates = ['metal', 'wood', 'plastic', 'cotton', 'glass', 'fabric', 'stone', 'rubber', 'ceramic',
                      'cloth', 'leather', 'flour', 'paper', 'clay', 'wax', 'concrete']
    elif subset == 'component':
        candidates = ['yes', 'no']
    elif subset == 'others_yes':
        candidates = ['yes', 'no']
    elif subset == 'others_number':
        candidates = ['2', '4', '6', '1', '8', '3', '5']
    elif subset == 'others_other':
        candidates = ['long', 'small', 'short', 'large', 'forest', 'water', 'ocean', 'big', 'tree', 'ground', 'tall',
                      'wild', 'outside', 'thin', 'head', 'thick', 'circle', 'brown', 'soft', 'land', 'neck', 'rough',
                      'chest', 'smooth', 'fur', 'hard', 'top', 'plants', 'black', 'metal', 'books', 'vertical', 'lake',
                      'grass', 'road', 'sky', 'front', 'kitchen', 'feathers', 'stripes', 'baby', 'hair', 'feet',
                      'mouth', 'female', 'table']
    else:
        print("Subset does not exist!")
        candidates = []
    return candidates


def load_prompt(question, idx=0):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question),
               ]
    return prompts[idx]


def load_demonstrations(subset, idx=0):
    df = pandas.read_csv('../datasets/dev/{}.csv'.format(subset), header=0)
    demonstrations = ""
    for i in range(len(df)):
        question = df['question'][i]
        answer = df['answer'][i]
        prompts = ["{} {}. ".format(question, answer),
                   "{} Answer: {}. ".format(question, answer),
                   "{} The answer is {}. ".format(question, answer),
                   "Question: {} Answer: {}. ".format(question, answer),
                   "Question: {} The answer is {}. ".format(question, answer),
                   ]
        demonstrations += prompts[idx]
    return demonstrations


def init_model(model_name="llama-7B"):
    tokenizer, model = None, None
    if model_name == "llama-30B" or model_name == "llama-65B":
        tokenizer = AutoTokenizer.from_pretrained("/home/xiaheming/data/pretrained_models/{}".format(model_name),
                                                  use_fast=False)
        model = LlamaForCausalLM.from_pretrained("/home/xiaheming/data/pretrained_models/{}".format(model_name),
                                                 torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    elif 'llama' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("/home/xiaheming/data/pretrained_models/{}".format(model_name),
                                                  use_fast=False)
        model = LlamaForCausalLM.from_pretrained("/home/xiaheming/data/pretrained_models/{}".format(model_name),
                                                 torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    elif "opt" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("/home/xiaheming/data/pretrained_models/{}".format(model_name),
                                                  use_fast=False)
        model = OPTForCausalLM.from_pretrained("/home/xiaheming/data/pretrained_models/{}".format(model_name)).to(device)
    else:
        print("Error: model not supported!!!\nSupported models: [llama]")
    return tokenizer, model


def get_start_loc(tokenizer, prompt):
    """the next position is the start location"""
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
    return tokens.shape[1]


@torch.no_grad()
def test(tokenizer, model, subset='color', model_name='llama-7B', prompt_idx=0, icl=False):
    cnt = 0
    correct = 0
    df = pandas.read_csv('../datasets/{}.csv'.format(subset), header=0)
    results = []
    id2scores = {}
    for i in tqdm(range(len(df))):
        cnt += 1
        sub_subset = None
        pred_scores = []  # prediction scores of candidates
        question = df['question'][i]
        answer = df['answer'][i]
        if subset == 'others':
            if answer in ['yes', 'no']:
                sub_subset = 'others_yes'
            elif answer in ['2', '4', '6', '1', '8', '3', '5']:
                sub_subset = 'others_number'
            else:
                sub_subset = 'others_other'
            candidates = load_candidates(sub_subset)
        else:
            candidates = load_candidates(subset)
        prompt = load_prompt(question, prompt_idx)
        if icl:
            if subset == 'others':
                demonstrations = load_demonstrations(sub_subset, idx=prompt_idx)
            else:
                demonstrations = load_demonstrations(subset, idx=prompt_idx)
            prompt = demonstrations + prompt
        start_loc = get_start_loc(tokenizer, prompt)
        for candidate in candidates:
            sent = prompt + " {}.".format(candidate)
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            labels = copy.deepcopy(inputs["input_ids"])
            labels[0, :start_loc] = -100
            labels[0, start_loc + 1:] = -100  # the location of the candidate
            loss = model(**inputs, labels=labels).loss
            pred_scores.append(loss.item())
        pred_scores = np.reciprocal(pred_scores)
        pred = candidates[np.argmax(pred_scores)]
        result_line = {"question_id": cnt, "answer": pred}
        id2scores[cnt] = pred_scores
        results.append(result_line)
        if pred == answer:
            correct += 1
    write_json(results, subset, model_name, prompt_idx=prompt_idx)
    pickle.dump(id2scores, open("./results/{}/{}/prompt{}.p".format(model_name, subset, prompt_idx), "wb"))
    print("Accuracy: ", correct / cnt)
    return correct / cnt


def eval(model_name, use_icl):
    subset_list = ['color']  # , 'shape', 'material', 'component', 'others'
    tokenizer, model = init_model(model_name)
    for subset in subset_list:
        print("Tested on the {} subset...".format(subset))
        results = []
        for idx in range(0, 5):  # prompt idx
            acc = test(tokenizer, model, subset=subset, model_name=model_name, prompt_idx=idx, icl=use_icl)
            results.append(acc)
        with open("./results/{}/{}/results.txt".format(model_name, subset), "w") as f:
            for idx, acc in enumerate(results):
                f.write("Accuracy for prompt{}: {} \n".format(idx, acc))
            avg = np.mean(results)
            std = np.std(results, ddof=1)  # np.sqrt(( a.var() * a.size) / (a.size - 1))
            f.write("Mean result: {}, Std result: {}".format(100 * avg, 100 * std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser of ImageNetVC')
    parser.add_argument('--model-name', default='llama-7b', type=str, help='Supported models: [llama-X, opt-X]')
    parser.add_argument('--use-icl', default=False, action='store_true', help='use in-context learning or not')
    args = parser.parse_args()
    eval(model_name=args.model_name, use_icl=args.use_icl)
