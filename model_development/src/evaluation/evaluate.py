import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from codebleu import calc_codebleu
import evaluate
import time

def evaluate_model(config):
    '''Evaluate trained model with multiple metrics'''
    
    print("Loading model for evaluation...")
    
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config['base_model'], torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, config['model_path'])
    model = model.merge_and_unload()
    model.eval()
    
    # Load eval dataset
    eval_dataset = load_dataset("json", data_files=config['eval_dataset_path'], split="train")
    
    predictions = []
    references = []
    
    print("Generating predictions...")
    for item in eval_dataset:
        inputs = tokenizer(item['prompt'], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=config['max_length'],
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        references.append(item['reference'])
    
    # Calculate metrics
    results = {}
    
    # BLEU
    bleu = evaluate.load("bleu")
    results['bleu'] = bleu.compute(predictions=predictions, references=[[r] for r in references])['bleu']
    
    # CodeBLEU
    try:
        codebleu_result = calc_codebleu(
            references=[[r] for r in references],
            predictions=predictions,
            lang=config.get('language', 'cpp')
        )
        results['codebleu'] = codebleu_result['codebleu']
    except:
        results['codebleu'] = None
    
    # Syntax validity
    import ast
    valid = sum(1 for pred in predictions if check_syntax(pred, config.get('language')))
    results['syntax_validity'] = valid / len(predictions)
    
    return results

def check_syntax(code, language):
    try:
        if language == 'python':
            ast.parse(code)
        return True
    except:
        return False