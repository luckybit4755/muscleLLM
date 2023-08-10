# Oh, yeah! 

So here's what's going on with this small langchain + llamacpp experiment: 

We're running some locally LLM's (LangChain chats) and have 2
personalities - Muscle Man and AI Assistant.

The main goal is to explore what LangChain can do by adding a local
document store and looking into its context-based features.

Anyone interested in submitting ideas or suggestions for improvement is
welcome to participate!

# Switching characters

Currently, there are just 2 personalities: Muscle Man and AI Assistant. 

1. ./muscleLLM.py -m models/StableBeluga-13B-GGML/stablebeluga-13b.ggmlv3.q2_K.bin 
2. ./muscleLLM.py -m [yada-yada] -p ai
3. ./muscleLLM.py -k http://localhost:5001/ -p rudy

The -m will use llama.cpp and the -k will use kolboldcpp

# Getting the stuff

## without gpu support

Just run: pip -r requirements.txt

## llama.cpp with gpu support

1. (optionally) change requirements.txt to use faiss-gpu instead of faiss-cpu
2. pip install -r requirements.txt
3. export CMAKE_ARGS="-DLLAMA_CUBLAS=on" ; export FORCE_CMAKE=1
4. pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

Be sure to pass --n_gpu_layers xx to muscleLLM.py

## kolboldcpp 

1. git clone https://github.com/LostRuins/koboldcpp/
2. change Makefile: NVCCFLAGS += -arch=native -> NVCCFLAGS += -arch=all
3. make LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 LLAMA_CUBLAS=1
3. ./koboldcpp.py --stream --contextsize 8192 --useclblast 0 0 --gpulayers 29 WizardCoder-15B-1.0.ggmlv3.q4_0.bin 

## llama.cpp with gpu support

Not really needed but pretty sweet

1. clone git@github.com:ggerganov/llama.cpp.git
2. edit the makefile change: NVCCFLAGS += -arch=all ; instead of arch=native
3. make clean && LLAMA_CUBLAS=1 make -j


## get models if needed

git clone git@hf.co:TheBloke/Llama-2-7B-Chat-GGML

idk of other sources for GGML models or if GPTQ models will work...

# ideas for more awesome:

1. research assistant
2. multi-actor stories

https://github.com/ShumzZzZz/GPT-Rambling/blob/main/LangChain%20Specific/langchain_persist_conversation_memory.ipynb

