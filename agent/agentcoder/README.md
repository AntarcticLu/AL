AgentCoder: Multi-Agent Code Generation with Effective Testing and Self-optimisation
[paper](https://arxiv.org/pdf/2312.13010)
[code](https://github.com/huangd1999/AgentCoder/tree/main)

first vllm start
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 --port 8080 --api-key token-abc123 \
--model /share_data/llm_weights/Qwen2.5-32B-Instruct --served-model-name model1 \
--tensor-parallel-size 4 --pipeline-parallel-size 2 \
--enable-auto-tool-choice --tool-call-parser hermes 
```
original 
```shell
sh original_agentcoder.sh
```
langgraph
```shell
python langgraph_agentcoder.py
```
