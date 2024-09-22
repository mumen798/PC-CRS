import sys

sys.path.append("..")

from src.model.KBRD import KBRD
from src.model.BARCOR import BARCOR
from src.model.UNICRS import UNICRS
from src.model.CHATGPT import CHATGPT
from src.model.PersuasiveCHATGPT import PersuasiveCHATGPT
from src.model.KGCHATGPT import KGCHATGPT
from src.model.MACRS import MACRS
from src.model.PersuasiveCHATGPT1 import PersuasiveCHATGPT1
from src.model.PersuasiveCHATGPT2 import PersuasiveCHATGPT2
from src.model.PersuasiveCHATGPT3 import PersuasiveCHATGPT3
# from src.model.PersuasiveLLAMA import PersuasiveLLAMA
# from src.model.MALLAMA import MALLAMA
from src.model.CHATLLAMA import CHATLLAMA
# from src.model.KGLLAMA import KGLLAMA

name2class = {
    'kbrd': KBRD,
    'barcor': BARCOR,
    'unicrs': UNICRS,
    'chatgpt': CHATGPT,
    'persuasive_chatgpt': PersuasiveCHATGPT,
    'kg_chatgpt': KGCHATGPT,
    'macrs': MACRS,
    'persuasive_chatgpt1': PersuasiveCHATGPT1,
    'persuasive_chatgpt2': PersuasiveCHATGPT2,
    'persuasive_chatgpt3': PersuasiveCHATGPT3,
    # 'persuasive_llama': PersuasiveLLAMA,
    # 'ma_llama': MALLAMA,
    'chat_llama': CHATLLAMA,
    # 'kg_llama': KGLLAMA
}


class RECOMMENDER():
    def __init__(self, crs_model, *args, **kwargs) -> None:
        model_class = name2class[crs_model]
        self.crs_model = model_class(*args, **kwargs)

    def get_rec(self, conv_dict):
        return self.crs_model.get_rec(conv_dict)

    def get_conv(self, conv_dict):
        return self.crs_model.get_conv(conv_dict)

    def get_choice(self, gen_inputs, option, state, conv_dict=None):
        return self.crs_model.get_choice(gen_inputs, option, state, conv_dict)
