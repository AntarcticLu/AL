#加载数据
from datasets import load_dataset
dataset=load_dataset("parquet",data_files='../../datasets/test-00000-of-00001.parquet',split='train')
dataset = [entry for entry in dataset]
#加载提示词
with open("./humaneval_prompt_update.txt", "r") as f:
    prompt1 = f.read()
with open("./test_designer_humaneval_prompt_update.txt", "r") as f:
    prompt2 = f.read()
from langchain_core.prompts import ChatPromptTemplate
prompt1=ChatPromptTemplate.from_messages([
    ("system","You are a software programmer."),
    ("human","\n"+prompt1+"""\n\n**Input Code Snippet**:\n```python\n{prompt}\n```\n## Completion 3:\n""")
])
prompt2=ChatPromptTemplate.from_messages([
    ("system","You are a code developer assistant."),
    ("human","\n"+prompt2+"""\n\n**Input Code Snippet**:\n```python\n{prompt}\n```\n""")
])
#去除代码多余字符串
def preprocess_data(completion_string):
    if f"```python" in completion_string:
        completion_string = completion_string[completion_string.find(f"```python")+len(f"```python"):]
        completion_string = completion_string[:completion_string.find("```")]
    else:
        print("Error: No code block found")
    return completion_string
#声明agent的LLM
from langchain_openai import ChatOpenAI
llm1 = prompt1 | ChatOpenAI(model="model1",
        api_key="token-abc123",
        base_url="http://localhost:8080/v1")
llm2 = prompt2 | ChatOpenAI(model="model1",
        api_key="token-abc123",
        base_url="http://localhost:8080/v1")
#开始创建图
from typing_extensions import TypedDict
class State(TypedDict):
    task_id : str     #任务id
    prompt : None     #题目，函数定义和变量，但没内容
    entry_point : str #函数名
    test : None       #测试代码，assert格式
    completion_list : None #5个补充代码的函数
    test_case_list : None  #10个大模型生成的测试用例
    need_reproduce : bool  #是否需要fix bug，需要就再生成代码，不需要就结束
    completion : None      #5个函数中最好的
    max_epoch : int        #不要一直fix bug，设置循环上限
#编程agent
def programmer_agent(data):
    completions_code = []
    for _ in range(5):
        while True:
            try:
                chat_response=llm1.invoke({"prompt":data["prompt"]})
                completion = chat_response.content
                completion = preprocess_data(completion)
            except Exception as e:
                print(e)
                completion = ""
            if completion!="":
                break
        completions_code.append(completion)
    return {"completion_list":completions_code}
#测试代码设计agent
def test_designer_agent(data):
    test_case_list = []
    for _ in range(10):
        while True:
            try:
                chat_response=llm2.invoke({"prompt":data["prompt"]})
                test_case = chat_response.content
                test_case = preprocess_data(test_case)
            except Exception as e:
                print(e)
                test_case = ""
            if test_case!="":
                break
        test_case_list.append(test_case)
    return {"test_case_list":test_case_list}
#代码测试agent
from untils import IMPORT_HELPER
from execution import check_correctness
test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
def test_res(data):
    completion_list=data["completion_list"]
    test_case_list=data["test_case_list"]
    correct_list = []
    for i in range(len(completion_list)):
        correct = 0
        if f"def {data['entry_point']}" not in completion_list[i]:
            correct_list.append(correct)
            continue
        for j in range(len(test_case_list)):
            if f"assert {data['entry_point']}(" not in test_case_list[j]:
                continue
            data["test_code"] = test_setup + "\n" + completion_list[i] + "\n" + test_case_list[j]
            data["generation"]=completion_list[i]
            result = check_correctness(data["task_id"], data, 'python', 3, "./tmp")
            if result["passed"]:
                correct += 1
        correct_list.append(correct)
    max_correct = max(correct_list)  #通过最大测试数
    idx = correct_list.index(max_correct)  #是哪个函数这么牛
    completion = data["completion_list"][idx]  #找到这个函数
    #只要对3个就不需要fix bug
    need_reproduce = max_correct<3
    return {"completion":completion,"need_reproduce":need_reproduce,
            "max_epoch":data["max_epoch"]-1}
#判断是否需要fix bug
def if_fix_bug(data):
    if data["max_epoch"]<0:
        return "False"
    return str(data["need_reproduce"])
#一些最终测试所需的函数，我也不懂
import io
import contextlib
import signal
class TimeoutException(Exception):
    pass
class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False
class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'
@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
#生成图逻辑
from langgraph.graph import StateGraph, START, END
agent_system  = (StateGraph(State)
                 .add_node("programmer", programmer_agent)        
                 .add_node("test_designer", test_designer_agent)  
                 .add_node("test_exec", test_res)                 
                 .add_edge(START, "programmer")                   
                 .add_edge("programmer", "test_designer")
                 .add_edge("test_designer", "test_exec")
                 .add_conditional_edges("test_exec", if_fix_bug, {"True":"programmer","False":END})
                 .compile()
                 )

#多线程执行程序
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
correct = 0
new_dataset=[]
with ThreadPoolExecutor(max_workers=5) as executor:
    new_data = [executor.submit(agent_system.invoke,{"task_id" : data["task_id"],"prompt" : data["prompt"], "test":data["test"],
                                        "entry_point" : data["entry_point"],"max_epoch" : 3}) for data in tqdm(dataset)]
    for datakey in tqdm(concurrent.futures.as_completed(new_data)):
        data=datakey.result()
        new_dataset.append(data)
#计算正确个数
for data in new_dataset:
    try:
        with swallow_io():
            with time_limit(2.0):
                exec(test_setup + "\n" + data["completion"] + "\n" + data["test"] + "\n" + f"check({data['entry_point']})")
            correct+=1
    except Exception as exc:
        pass
#输出结果
print("==============Start Report Testing==============")
print(f"test_report: {(correct/len(dataset)*100):.1f}")