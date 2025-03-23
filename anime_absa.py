from torch.utils.checkpoint import checkpoint

from InstructABSA.InstructABSA.utils import T5Generator
from anime_absa_config import Config
from InstructABSA.instructions import InstructionsHandler

config = Config()
config.inst_type = 1
config.mode = 'ate'
config.model_checkpoint = 'kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined'
config.test_input = """iamjoe
Recommended
Death Note is original, awesome, and a great anime to watch. The only reason it isn't perfect is because of Near who has to be the lamest enemy ever. Also, the ending was a total cop-out. DUMB, DUMB, DUMB.

The art wasn't that great, but I sure did like this anime anyway.

The sound was fitting; it suited the tense atmosphere and scenes.

The characters were all intriguing and deep, especially Light. It's cool to see such an evil protagonist in an anime. He's certainly original.

If you want a super-cool anime to watch, watch Death Note. Except for L and Near, who made everything totally unfabulous.
Reviewer’s Rating: 9"""
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
aspect_list = [word.strip() for word in aspect_term_decoded.split(',')]
print('Model output: ', aspect_term_decoded)