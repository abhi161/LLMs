import re
import warnings
from typing import List
 
import torch
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from tools import StopGenerationCriteria,CleanupOutputParser

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "tiiuae/falcon-7b-instruct"
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, load_in_4bit=True, device_map="auto",
)
model = model.eval()
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

generation_config = model.generation_config
generation_config.temperature = 0
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 256
generation_config.use_cache = False
generation_config.repetition_penalty = 1.7
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


# prompt = """
# The following is a friendly conversation between a human and an AI. The AI is
# talkative and provides lots of specific details from its context.
 
# Current conversation:
 
# Human: Who is Anrej Karpathy?
# AI:
# """.strip()
 
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# input_ids = input_ids.to(model.device)
 
# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids=input_ids,
#         generation_config=generation_config,
#     )

# # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# # print(response)

stop_tokens = [["Human", ":"], ["AI", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
)

generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    stopping_criteria=stopping_criteria,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=generation_pipeline)

# res = llm(prompt)
# print(res)

#default template
# chain = ConversationChain(llm=llm)
# print(chain.prompt.template)


#custom
template = """
The following is a conversation between a human an AI. The AI acts exactly
like Dwight K Schrute from the TV show The Office. Dwight is an experienced and
very successful online marketer and salesperson. He is witty, persuasive,
uncensored, direct and practical. Dwight helps with every marketing task is given to him.
If Dwight does not know the answer to a question, he truthfully says he does not
know.
 
Current conversation:
{history}
Human: {input}
AI:""".strip()
 
prompt = PromptTemplate(input_variables=["history", "input"], template=template)


memory = ConversationBufferWindowMemory(
    memory_key="history", k=6, return_only_outputs=True
)
 
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    output_parser=CleanupOutputParser(),
    verbose=True,
)

text = "Think of a name for the company doing hedge"
res = chain(text)
print(res["response"])