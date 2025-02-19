import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
CACHE_DIR = "/shared/4/models"
#MODEL_DIR = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR = "meta-llama/Llama-3.3-70B-Instruct"

NUM_GPUS = 4

import json
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm.auto import tqdm

import pandas as pd
import math

# Optionally, you could render input into the following template by yourself. `tokenizer.apply_chat_template`` is a simple way of doing this.
# llama3_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

print('reading data')
df = pd.read_csv('/shared/4/projects/reddit-morals/data/moral-questions.csv', dtype=str)

# Render input text into llm format
print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, cache_dir=CACHE_DIR, trust_remote_code=True)

# Create a model instance
print('Loading model')
model = LLM(model=MODEL_DIR, download_dir=CACHE_DIR, tensor_parallel_size=NUM_GPUS, max_model_len=4096*2)


# split into chunk to process a few at a time

#outf = open('/shared/4/projects/reddit-morals/data/moral-dilemmas.llama3.1-8b.paraphrase.tsv', 'wb')
outf = open('/shared/4/projects/reddit-morals/data/moral-dilemmas.llama3.3-70b.paraphrase.tsv', 'wb')
outf.write('id\tsummary\toption1\toption2\n'.encode('utf-8'))

ex1 = {
    "summary": "A woman was on her deathbed. There was one drug that the doctors said would save her. It was a form of radium that a druggist in the same town had recently discovered. The drug was expensive to make, but the druggist was charging ten times what the drug cost him to produce. He paid $200 for the radium and charged $2,000 for a small dose of the drug. The sick woman's husband, Heinz, went to everyone he knew to borrow the money, but he could only get together about $1,000 which is half of what it cost. He told the druggist that his wife was dying and asked him to sell it cheaper or let him pay later. But the druggist said: \"No, I discovered the drug and I'm going to make money from it.” What is the correct moral action for Heinz to take?",
    'option1': 'Steal the drug',
    'option2': "Don't steal the drug and let his wife die"
}
ex2 = {
    "summary": "A person cheated on their boyfriend while at a conference, which occurred right after a huge fight where they both said they wanted to break up. However, now that they’re back together, the question becomes: should the boyfriend be told?",
    'option1': 'Yes, tell the boyfriend',
    'option2': "No, don't tell the boyfriend"
}
ex3 = {
    "summary": "A prestigious school has one remaining space to admit a student. The headmaster must choose between a talented student from a disadvantaged background or a student with excellent academic records but from a well-off family. Which student should the headmaster choose?",
    "option1": "the talented student from a disadvantaged background",
    "option2": "the student with excellent academic records from a well-off family",
}
chunks = np.array_split(df, math.ceil(len(df) / 1000))
for ci, chunk in enumerate(tqdm(chunks)):

    input_list = []
    # Inputs (A list of conversations)
    print('Generating input list')
    for j, row in chunk.iterrows():

        question = row['cleaned_question']
        instruction = 'The following text is a question about what is the correct moral decision to take in a situation:\n' \
        + 'Question:\n%s\n' \
        + '\nInstructions:\n' \
        + ' Paraphase the question as a general moral dilemma where a person has to choose between two mutually-exclusive options that each describe a specific action to take.\n' \
        + ' Generate an answer in JSON format with the following information:\n' \
        + ' * A short summary of the question as a moral dilemma. The summary should describe the specific setting and highlight the general moral dilemma that the author faces. This summary should have the JSON key "summary".\n' \
        + ' * A one sentence description of the first action that could be taken based on the dilemma, with the JSON key "option1".\n' \
        + ' * A one sentence description of the second action that could be taken based on the dilemma, with the JSON key "option2".\n' \
        + 'The two actions should describe distinct choices that can be taken based on the dilemma. Options should reflect different values or morals. Actions should be mutually exclusive and have distinct outcomes.\n' \
        + '\nExample of paraphrased scenarios and options:\n' \
        + json.dumps(ex1) + '\n' \
        + json.dumps(ex2) + '\n' \
        + json.dumps(ex3) + '\n' \
        + 'Generate your paraphrase of the first question as a moral dilemma like the examples. ' \
        + 'Do not explain your answer. Start your response with the text "Answer: " before the JSON.\n'

        instruction = instruction % question



        input_list.append(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ]
        )

    #
    print('Processsing chunk %d' % ci)
    input_list = [tokenizer.apply_chat_template(user_input, tokenize=False,add_special_tokens=False, add_generation_prompt=True) for user_input in input_list]


    print('Generating output')
    # Model generation
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
    output_list = model.generate(input_list, sampling_params=sampling_params)

    # Extract model response text
    output_list = [output.outputs[0].text.strip() for output in output_list]
    for i, ans in enumerate(output_list):

        post_id = chunk.iloc[i]['id']

        if 'Answer:' in ans:
            idx = ans.index('Answer:')
            ans = ans[idx + len('Answer:'):].strip()
            ans = ' '.join(ans.split())
        try:
            j = json.loads(ans)
            summary = j.get('summary', '')
            opt1 = j.get('option1', '')
            opt2 = j.get('option2', '')
        except:
            summary = ''
            opt1 = ''
            opt2 = ''

        #print(ans)
        output = '%s\t%s\t%s\t%s\n' % (post_id, summary, opt1, opt2)
        output = output.encode('utf-8')
        outf.write(output)
    outf.flush()
