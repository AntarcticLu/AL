from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy
import concurrent
import json
prompt_path = "./humaneval_prompt_update.txt"
with open(prompt_path, "r") as f:
    construct_few_shot_prompt = f.read()
client=None
def preprocess_data(completion_string):
    if f"```python" in completion_string:
        completion_string = completion_string[completion_string.find(f"```python")+len(f"```python"):]
        completion_string = completion_string[:completion_string.find("```")]
    else:
        print("Error: No code block found")
    return completion_string
def fetch_completion(data_entry ,times = 5):
    global construct_few_shot_prompt,client
    if "need_reproduce" in data_entry.keys() and data_entry["need_reproduce"]==False:
        return data_entry
    prompt = data_entry["prompt"]
    text = f"""
{construct_few_shot_prompt}

**Input Code Snippet**:
```python
{prompt}
```
## Completion 3:
"""
    completions_code = []
    for i in range(times):
        while True:
            try:
                chat_response = client.chat.completions.create(
                    model="model1",
                    messages=[
                        {"role": "system", "content": "You are a software programmer."},
                        {"role": "user", "content":text},
                        ]
                )
                completion = chat_response.choices[0].message.content
                completion = preprocess_data(completion)
            except Exception as e:
                print(e)
                completion = ""
            if completion!="":
                break
        completions_code.append(completion)
    data_entry["completion_list"] = completions_code
    return data_entry
def call_fetch_completion_helper(dataset,model):
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
if __name__ == "__main__":
    dataset=load_dataset("parquet",data_files='../../datasets/test-00000-of-00001.parquet',split='train')
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
    with open(f"./model_python.json", "w") as f:
        json.dump(dataset, f, indent=4)