import openpyxl
import torch
from bayeformers import to_bayesian
from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from scipy.special import softmax
import splogger as log

SAMPLES = 10

def setup_model(model_name: str, lower_case: bool):
    config    = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    return model, tokenizer

def generate_input(data):
    i = 0
    while True:
        i += 1
        if i == 1:
            continue
        yield data[f'A{i}'].value, data[f'B{i}'].value

if __name__ == '__main__':
    print("== Loading model")
    model, tokenizer = setup_model('distilbert-base-uncased', True)
    model = to_bayesian(model, delta = 0.1, skip_n_firsts = 86).cuda()
    print("== Loading dict")
    model.load_state_dict(torch.load("b_model_untrained.pth"))

    print("== Loading dataset")
    data = generate_input(openpyxl.load_workbook(filename = "/home/win32gg/Downloads/cats eval.xlsx")['Sheet1']) # Aint nobody got time for args
    for example in data:
        # Prepare data
        if None in example:
            break # end
        inputs = tokenizer(*example, max_length=512)
        inputs['input_ids'] = torch.tensor(inputs['input_ids']).unsqueeze(0).cuda()
        inputs['attention_mask'] = torch.tensor(inputs['attention_mask']).unsqueeze(0).cuda()
        
        outputs = model(**inputs)
        
        start_probas = softmax(outputs.start_logits.cpu().detach().numpy())
        end_probas   = softmax(outputs.end_logits.cpu().detach().numpy())

        start_pos    = start_probas.argmax()
        end_pos      = end_probas.argmax()

        # print("== QUESTION ==")
        # print(example)
        # print("=== ANSWER ===")
        if start_pos > end_pos:
            print("[NO ANSWER]")
            continue
        # print(start_pos, end_pos)
        print(tokenizer.decode(inputs['input_ids'].squeeze()[start_pos:end_pos]))