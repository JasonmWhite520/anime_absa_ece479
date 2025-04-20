from sympy.codegen.ast import continue_
from torch.utils.checkpoint import checkpoint

from InstructABSA.InstructABSA.utils import T5Generator, T5Classifier
from anime_absa_config import Config
from InstructABSA.instructions import InstructionsHandler

def aspect_term_extraction(raw_review):

    config = Config()
    config.inst_type = 1
    config.mode = 'ate'
    config.model_checkpoint = 'kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined'
    config.test_input = raw_review
    instruct_handler = InstructionsHandler()
    model_checkpoint = config.model_checkpoint
    ood_tr_data_path = config.ood_tr_data_path
    ood_te_data_path = config.ood_te_data_path

    if config.inst_type == 1:
        instruct_handler.load_instruction_set1()
    else:
        instruct_handler.load_instruction_set2()

    if config.set_instruction_key == 1:
        indomain = 'bos_instruct1'
        outdomain = 'bos_instruct2'
    else:
        indomain = 'bos_instruct2'
        outdomain = 'bos_instruct1'
    #list_noaspectterm = []
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.ate[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ate[outdomain]
    eos_instruction = instruct_handler.ate['eos_instruct']

    if config.task == 'atsc':
        config.test_input, aspect_term = config.test_input.split('|')[0], config.test_input.split('|')[1]
        model_input = bos_instruction_id + config.test_input + f'. The aspect term is: {aspect_term}' + eos_instruction
    else:
        model_input = bos_instruction_id + config.test_input + eos_instruction

    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length = config.max_token_length)
    aspect_term_decoded = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(aspect_term_decoded)
    aspect_list = [word.strip() for word in aspect_term_decoded.split(',')]
    if "noaspectterm" in aspect_term_decoded:
        print("noaspectterm found")
        return []

    print('Extraction Model output: ', aspect_term_decoded)
    return aspect_list

def aspect_term_sentiment_classification(raw_review, aspect_term):
    config = Config()
    config.inst_type = 1
    config.mode = 'atsc'
    config.model_checkpoint = 'kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined'
    config.test_input = raw_review
    instruct_handler = InstructionsHandler()
    model_checkpoint = config.model_checkpoint
    ood_tr_data_path = config.ood_tr_data_path
    ood_te_data_path = config.ood_te_data_path

    if config.inst_type == 1:
        instruct_handler.load_instruction_set1()
    else:
        instruct_handler.load_instruction_set2()

    if config.set_instruction_key == 1:
        indomain = 'bos_instruct1'
        outdomain = 'bos_instruct2'
    else:
        indomain = 'bos_instruct2'
        outdomain = 'bos_instruct1'

    t5_exp = T5Classifier(model_checkpoint)
    bos_instruction_id = instruct_handler.atsc[indomain]
    eos_instruction = instruct_handler.atsc['eos_instruct']

    model_input = bos_instruction_id + config.test_input + f'. The aspect term is: {aspect_term}' + eos_instruction

    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length=config.max_token_length)
    aspect_term_sentiment_decoded = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #aspect_list = [word.strip() for word in aspect_term_decoded.split(',')]
    print('Sentiment Model output: ', aspect_term_sentiment_decoded)
    return aspect_term_sentiment_decoded

def get_overall_review_sentiment(raw_review):
    aspect_list = aspect_term_extraction(raw_review)
    if len(aspect_list) == 0:
        return [None, None]
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
        return [None, None]
    for sentiment in clean_aspect_sentiments:
        if sentiment == "positive":
            total_sentiment += 1
        elif sentiment == "conflict" or sentiment == "neutral":
            total_sentiment += 0.5

    avg_sentiment = total_sentiment / len(clean_aspect_sentiments)
    return avg_sentiment, aspect_sentiment_mapping