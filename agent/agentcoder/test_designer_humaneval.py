from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import copy
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import json
with open("./test_designer_humaneval_prompt_update.txt", "r") as f:
    construct_few_shot_prompt = f.read()
client=None
def preprocess_data(test_case_string):
    if f"```python" in test_case_string:
        test_case_string = test_case_string[test_case_string.find(f"```python")+len(f"```python"):]
        test_case_string = test_case_string[:test_case_string.find("```")]
    return test_case_string
def fetch_completion(data_entry,times=10):
    global construct_few_shot_prompt,client
    if "need_reproduce" in data_entry.keys() and data_entry["need_reproduce"]==False:
        return data_entry
    prompt = data_entry["prompt"]
    entry_point = data_entry["entry_point"]
    
    text = f"""
{construct_few_shot_prompt}

**Input Code Snippet**:
```python
{prompt}
```
"""
    test_case_list = []
    for i in range(times):
        while True:
            try:
                chat_response = client.chat.completions.create(
                model="model1",
                messages=[
                    {"role": "system", "content": "You are a code developer assistant."},
                    {"role": "user", "content":text},
                    ]
                )
                test_case = chat_response.choices[0].message.content
                test_case = preprocess_data(test_case)
            except Exception as e:
                print(e)
                test_case = ""
            if test_case!="":
                break
        test_case_list.append(test_case)
    data_entry["test_case_list"] = test_case_list
    return data_entry
def call_fetch_test_completion_helper(dataset,model):
    print("Fixing bug...")
    global client
    client=model
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_entry = {executor.submit(fetch_completion, copy.deepcopy(entry)): entry for entry in tqdm(dataset)}
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))
    return dataset

if __name__=="__main__":
    with open(f"./model_python.json", "r") as f:
        dataset = json.load(f)
    dataset = [entry for entry in dataset]
    openai_api_key = "token-abc123"
    openai_api_base = "http://localhost:8080/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_entry = {executor.submit(fetch_completion, copy.deepcopy(entry)): entry for entry in tqdm(dataset)}
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))
    with open(f"./model_python2.json", "w") as f:
        json.dump(dataset, f, indent=4)