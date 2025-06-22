import dotenv, os
import openai

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Reload the utils module to get the updated version
import importlib

import lllm.utils
importlib.reload(lllm.utils)

# Reload the questions_loaders module to ensure it uses the updated utils
import importlib
import lllm.questions_loaders
importlib.reload(lllm.questions_loaders)
from lllm.questions_loaders import Sciq

dataset = Sciq()
dataset.head()

number_questions_to_answer = 60
dataset.check_if_model_can_answer(
    model="gpt-3.5-turbo",
    max_questions_to_try=number_questions_to_answer,  # for trying only a few
    max_batch_size=20,
    save_progress=True,
    bypass_cost_check=True,
    # if False, the code provides an estimate of the API cost and asks for confirmation before proceeding
    regenerate_if_done_before=True,  # if True, it will overwrite the results. Use with care. 
    model_kwargs={
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 64,
        "top_p": 1,
    },
)


number_questions_to_answer = 20
dataset.does_model_lie(
    model="gpt-3.5-turbo",
    max_questions_to_try=number_questions_to_answer,
    lie_instructions=dataset.lie_instructions[0:2],
    # take only the first two instructions in the default list. If None is left, all are used
    test_questions=None,  # same here, but for the double down questions
    max_batch_size=20,
    question_prefix=None,
    answer_prefix=None,
    save_progress=True,
    bypass_cost_check=True,
    regenerate_if_done_before=False,
    model_kwargs={
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 128,
        "top_p": 1,
    },
)


import pandas as pd

inv_questions = pd.read_csv("data/probes.csv")["probe"].tolist()
inv_questions


number_questions_to_answer = 40
dataset.generate_false_statements(
    model="gpt-3.5-turbo",
    max_questions_to_try=number_questions_to_answer,
    save_progress=True,
    bypass_cost_check=True,
    regenerate_if_done_before=False,
    model_kwargs={
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 128,
        "top_p": 1,
    })

print(type(dataset.iloc[[0]]))  # likely <class 'pandas.core.frame.DataFrame'>
print(hasattr(dataset, 'generate_logprobs'))  # True
print(hasattr(dataset.iloc[[0]], 'generate_logprobs'))  # False


number_questions_to_answer = 8
dataset.generate_logprobs(
    model_suspect="gpt-3.5-turbo",
    max_questions_to_try=number_questions_to_answer,
    lie_instructions=dataset.lie_instructions[0:2],
    # in the actual experiments, we leave this to None, which means that all instructions are used
    truth_instructions=dataset.truth_instructions[0:2],  # same here
    lie_double_down_rate_threshold=0.8,
    oversample_cot=True,  # this makes sure 50% of the sampled instructions contain CoT
    save_progress=True,
    regenerate_if_done_before=True,
    model_kwargs_suspect={
        "temperature": 0.7,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 256,
        "top_p": 1,
        "stop": ["\n", "END"],
    },
)