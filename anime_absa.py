from sympy.codegen.ast import continue_
from torch.utils.checkpoint import checkpoint
import torch
from InstructABSA.InstructABSA.utils import T5Generator, T5Classifier
from anime_absa_config import Config
from InstructABSA.instructions import InstructionsHandler

atsc_config = Config()
atsc_config.inst_type = 1
atsc_config.mode = 'atsc'
atsc_config.model_checkpoint = 'kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined'
atsc_instruct_handler = InstructionsHandler()
atsc_model_checkpoint = atsc_config.model_checkpoint

if atsc_config.inst_type == 1:
    atsc_instruct_handler.load_instruction_set1()
else:
    atsc_instruct_handler.load_instruction_set2()

if atsc_config.set_instruction_key == 1:
    indomain = 'bos_instruct1'
    outdomain = 'bos_instruct2'
else:
    indomain = 'bos_instruct2'
    outdomain = 'bos_instruct1'

atsc_t5_exp = T5Classifier(atsc_model_checkpoint)
atsc_bos_instruction_id = atsc_instruct_handler.atsc[indomain]
atsc_eos_instruction = atsc_instruct_handler.atsc['eos_instruct']

ate_config = Config()
ate_config.inst_type = 1
ate_config.mode = 'ate'
ate_config.model_checkpoint = 'kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined'
ate_instruct_handler = InstructionsHandler()
ate_model_checkpoint = ate_config.model_checkpoint
ate_ood_tr_data_path = ate_config.ood_tr_data_path
ate_ood_te_data_path = ate_config.ood_te_data_path

if ate_config.inst_type == 1:
    ate_instruct_handler.load_instruction_set1()
else:
    ate_instruct_handler.load_instruction_set2()

if ate_config.set_instruction_key == 1:
    indomain = 'bos_instruct1'
    outdomain = 'bos_instruct2'
else:
    indomain = 'bos_instruct2'
    outdomain = 'bos_instruct1'
# list_noaspectterm = []
ate_t5_exp = T5Generator(ate_model_checkpoint)
ate_bos_instruction_id = ate_instruct_handler.ate[indomain]
if ate_ood_tr_data_path is not None or ate_ood_te_data_path is not None:
    bos_instruction_ood = ate_instruct_handler.ate[outdomain]
ate_eos_instruction = ate_instruct_handler.ate['eos_instruct']
def aspect_term_extraction(raw_review):


    ate_config.test_input = raw_review

    ate_model_input = ate_bos_instruction_id + ate_config.test_input + ate_eos_instruction

    input_ids = ate_t5_exp.tokenizer(ate_model_input, return_tensors="pt").input_ids
    outputs = ate_t5_exp.model.generate(input_ids, max_length = ate_config.max_token_length)
    aspect_term_decoded = ate_t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(aspect_term_decoded)
    aspect_list = [word.strip() for word in aspect_term_decoded.split(',')]
    if "noaspectterm" in aspect_term_decoded:
        print("noaspectterm found")
        return []

    print('Extraction Model output: ', aspect_term_decoded)
    return aspect_list

def aspect_term_sentiment_classification(raw_review, aspect_term):
    atsc_config.test_input = raw_review
    model_input = atsc_bos_instruction_id + atsc_config.test_input + f'. The aspect term is: {aspect_term}' + atsc_eos_instruction
    input_ids = atsc_t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = atsc_t5_exp.model.generate(input_ids, max_length=atsc_config.max_token_length)
    aspect_term_sentiment_decoded = atsc_t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #aspect_list = [word.strip() for word in aspect_term_decoded.split(',')]
    print('Sentiment Model output: ', aspect_term_sentiment_decoded)
    return aspect_term_sentiment_decoded

def get_overall_review_sentiment(raw_review):
    aspect_list = aspect_term_extraction(raw_review)
    if len(aspect_list) == 0:
        return None
    aspect_sentiments = []
    aspect_sentiment_mapping = []
    for aspect in aspect_list:
        aspect_term_sentiment = aspect_term_sentiment_classification(raw_review, aspect)
        aspect_sentiments.append(aspect_term_sentiment)
        aspect_sentiment_mapping.append({
            'aspect': aspect,
            'sentiment': aspect_term_sentiment
        })
    total_sentiment = 0
    clean_aspect_sentiments = [aspect for aspect in aspect_sentiments if aspect != 'none']
    if len(clean_aspect_sentiments) == 0:
        return None

    if len(clean_aspect_sentiments) == 0:
        return [None, None]
    for sentiment in clean_aspect_sentiments:
        if sentiment == "positive":
            total_sentiment += 1
        elif sentiment == "conflict" or sentiment == "neutral":
            total_sentiment += 0.5

    avg_sentiment = total_sentiment / len(clean_aspect_sentiments)
    return avg_sentiment, aspect_sentiment_mapping