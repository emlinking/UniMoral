import os
import torch
import tqdm
from vllm import LLM, SamplingParams

CACHE_DIR = "/shared/0/projects/code-switching/datasets"

def get_response(model_str, model, tokenizer, inputs, temperature, top_p, top_k, max_tokens):
    if model_str == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        return get_response_deepseek_distill(model, tokenizer, inputs, temperature, top_p, top_k, max_tokens)
    elif model_str == "microsoft/Phi-4-mini-reasoning":
        return get_response_phi(model, tokenizer, inputs, temperature, top_p, top_k, max_tokens)
    else:
        raise ValueError("Model not supported: {}".format(model_str))
    
def get_response_phi(model, tokenizer, inputs, temperature, top_p, top_k, max_tokens):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)

    conversations = []
    prompts = []

    print("Preparing inputs for vLLM")
    for input in tqdm.tqdm(inputs):
        conversation = []
        
        user_message = {"role": "user", "content": input}
        conversation.append(user_message)

        conversations.append(conversation)

        prompt = tokenizer.apply_chat_template(
                            conversation,
                            tokenize=False,)        
        prompts.append(prompt)

    # see here for default chat template: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/tokenizer_config.json
    return (prompts, model.chat(
            conversations,
            sampling_params,
    ))

def get_response_deepseek_distill(model, tokenizer, inputs, temperature, top_p, top_k, max_tokens):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)

    conversations = []
    prompts = []

    print("Preparing inputs for vLLM")
    for input in tqdm.tqdm(inputs):
        conversation = []
        
        user_message = {"role": "user", "content": input}
        conversation.append(user_message)

        conversations.append(conversation)

        prompt = tokenizer.apply_chat_template(
                            conversation,
                            tokenize=False,)        
        prompts.append(prompt)

    # see here for default chat template: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/tokenizer_config.json
    return (prompts, model.chat(
            conversations,
            sampling_params,
    ))

def load_model(model):
    print("Loading model: ", model)
    print("Using the following GPUs: ", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    # https://docs.vllm.ai/en/latest/api/vllm/config.html#vllm.config.ModelConfig.max_model_len
    return LLM(model=model, download_dir=CACHE_DIR, tensor_parallel_size=num_gpus)