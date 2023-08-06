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

So, the personality bits are stored in subdirectories with multiple .txt files because I'm kind of lazy and it's convenient for me. 

Be sure to run in the same environment you build lama-cpp-python etc

You can contribute if you want!

# Getting the stuff

## build llama.cpp with gpu support

1. clone git@github.com:ggerganov/llama.cpp.git
2. edit the makefile change: NVCCFLAGS += -arch=all ; instead of arch=native
3. make clean && LLAMA_CUBLAS=1 make -j

## build lama-cpp-python

1.  git clone git@github.com:abetlen/llama-cpp-python.git
2.  pyenv virtualenv llama_cpp_python
3.  pyenv activate llama_cpp_python
4.  python -m pip install --upgrade pip
5.  pip install scikit-build cmake
6.  python setup.py clean && python setup.py install && echo 'my mom!'

## get models if needed

1. git clone git@hf.co:TheBloke/Llama-2-7B-Chat-GGML

## even more dependencies

1. pip install langchain sentence_transformers faiss-gpu
3. ./muscleLLM.py -m models/StableBeluga-13B-GGML/stablebeluga-13b.ggmlv3.q2_K.bin
3. ./muscleLLM.py -m models/StableBeluga-13B-GGML/stablebeluga-13b.ggmlv3.q2_K.bin -p ai 

