import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument("--video_input",help="video path")
    parser.add_argument("--image_input",help="image path")
    parser.add_argument("--prompt", default="Your task is to determine if there is any abnormal movement in the video", 
                        help="Ask a question")
    parser.add_argument("--num_beams", default=1, help="beam search numbers [1,10]")
    parser.add_argument("--temperature", default=1.0, help="Temperature [0.1,2.0]")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================
if __name__== "__main__":
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    query = args.prompt
    video = args.video_input
    image = args.image_input
    num_beams = args.num_beams
    temperature = args.temperature

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    chatbot = []
    print('Initialization Finished')

# ========================================
#             Inference Setting
# ========================================

# def reset(chat_state, img_list):
#     if chat_state is not None:
#         chat_state.messages = []
#     if img_list is not None:
#         img_list.clear()
#         return None, None, None, 'Please upload your video first', 'Upload & Start Chat', chat_state, img_list


def upload_video(gr_video, gr_img, text_input, chat_state,chatbot):
    if chatbot is None:
        chatbot = []
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    if gr_img is None and gr_video is None:
        return None, None, None, True, chat_state, None
    elif gr_img is not None and gr_video is None:
        print(gr_img)
        chatbot = chatbot + [((gr_img,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        return None, None, None, None, chat_state, img_list,chatbot
    elif gr_video is not None and gr_img is None:
        print(gr_video)
        # chatbot.append(((gr_video,), None))
        chatbot =  chatbot + [((gr_video,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return None,None, None, None, chat_state, img_list, chatbot
    else:
        # img_list = []
        return None, None, None, chat_state, None,chatbot

def ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return None, 'Input should not be empty', chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot.append([user_message, None])
    return '', chatbot, chat_state


def answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list

### The guy's code
if args.model_type == 'vicuna':
    chat_state = default_conversation.copy()
else:
    chat_state = conv_llava_llama_2.copy()

video_path = "/home/student.aau.dk/fd27ct/projects/VAD/Video-LLaMA/examples/dog_barking.mp4"
chat_state.system = ""
img_list = []
llm_message = chat.upload_video(video_path , chat_state, img_list)

while True:
    user_message = input("User/ ")

    chat.ask(user_message, chat_state)

    num_beams = 2
    temperature = 1.0

    llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
    print(chat_state.get_prompt())
    print(chat_state)
    print(llm_message)