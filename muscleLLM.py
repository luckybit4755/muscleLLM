#!/usr/bin/env python
#############################################################################

import argparse
import logging
import os

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from utils import read_txt_files

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    BaseMemory,
    HumanMessage,
    SystemMessage,
)

import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS
from datetime import datetime
from langchain.memory import VectorStoreRetrieverMemory

class MuscleLLM:
    LOG = logging.getLogger(__name__)
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="my cool script is cool")
        self.parser.add_argument("--log_directory", type=str, default="logs")

        personality = self.parser.add_argument_group("personality")
        personality.add_argument("--personality", type=str, default="muscle-man")
        personality.add_argument("--personality_directory", type=str, default="personality")
        personality.add_argument("--index_directory", type=str, default="index")

        model = self.parser.add_argument_group("model")
        model.add_argument("--model_path",         type=str,        required=True  , help="Path to model to load")
        model.add_argument("--cache",              type=bool,       default=None)
        model.add_argument("--echo",               type=bool,       default=False  , help="Whether to echo the prompt.")
        model.add_argument("--f16_kv",             type=bool,       default=True   , help="Use half-precision for key/value cache.")
        model.add_argument("--last_n_tokens_size", type=int,        default=64     , help="The number of tokens to look back when applying the repeat_penalty.")
        model.add_argument("--logits_all",         type=bool,       default=False  , help="Return logits for all tokens, not just the last token.")
        model.add_argument("--logprobs",           type=int,        default=None   , help="The number of logprobs to return. If None, no logprobs are returned.")
        model.add_argument("--lora_base",          type=str,        default=None   , help="The path to the Llama LoRA base model.")
        model.add_argument("--lora_path",          type=str,        default=None   , help="The path to the Llama LoRA. If None, no LoRa is loaded.")
        model.add_argument("--max_tokens",         type=int,        default=256    , help="The maximum number of tokens to generate.")
        model.add_argument("--n_batch",            type=int,        default=512    , help="Number of tokens to process in parallel. Should be a number between 1 and n_ctx.")
        model.add_argument("--n_ctx",              type=int,        default=1024   , help="Token context window.")
        model.add_argument("--n_gpu_layers",       type=int,        default=32     , help="Number of layers to be loaded into gpu memory. Default 32.")
        model.add_argument("--n_parts",            type=int,        default=-1     , help="Number of parts to split the model into. If -1, the number of parts is automatically determined.")
        model.add_argument("--n_threads",          type=int,        default=None   , help="Number of threads to use. If None, the number of threads is automatically determined.")
        model.add_argument("--repeat_penalty",     type=float,      default=1.1    , help="The penalty to apply to repeated tokens.")
        model.add_argument("--rope_freq_base",     type=float,      default=10000.0, help="Base frequency for rope sampling.")
        model.add_argument("--rope_freq_scale",    type=float,      default=1.0    , help="Scale factor for rope sampling.")
        model.add_argument("--seed",               type=int,        default=-1     , help="Seed. If -1, a random seed is used.")
        model.add_argument("--stop",               type=str,        default=None   , help="A list of strings to stop generation when encountered. Comma separated list.")
        model.add_argument("--streaming",          type=bool,       default=True   , help="Whether to stream the results, token by token.")
        model.add_argument("--suffix",             type=str,        default=None   , help="A suffix to append to the generated text. If None, no suffix is appended.")
        model.add_argument("--tags",               type=str,        default=None   , help="Tags to add to the run trace. Comma separated list.")
        model.add_argument("--temperature",        type=float,      default=0.8    , help="The temperature to use for sampling.")
        model.add_argument("--top_k",              type=int,        default=40     , help="The top-k value to use for sampling.")
        model.add_argument("--top_p",              type=float,      default=0.95   , help="The top-p value to use for sampling.")
        model.add_argument("--use_mlock",          type=bool,       default=False  , help="Force system to keep model in RAM.")
        model.add_argument("--use_mmap",           type=bool,       default=True   , help="Whether to keep the model loaded in RAM")
        model.add_argument("--verbose",            type=bool,       default=True   , help="Print verbose output to stderr.")
        model.add_argument("--vocab_only",         type=bool,       default=False  , help="Only load the vocabulary, no weights.")
        # todo: param model_kwargs         Dict[str, Any] [Optional] :: Any additional parameters to pass to llama_cpp.Llama.
        # todo : model.add_argument("--metadata            type=                          Dict[str, Any] = None                                          Metadata to add to the run trace.
        model.add_argument("--embeddings_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    def main(self):
        args = self.parser.parse_args()
        personality = read_txt_files( f'{args.personality_directory}/{args.personality}' )

        llm = self.createLLM(args)

        ########################################################################
        # index....

        embeddings = HuggingFaceEmbeddings(model_name=args.embeddings_model_name)

        # https://python.langchain.com/docs/modules/memory/types/vectorstore_retriever_memory
        if os.path.exists(f'{args.index_directory}/{args.personality}.faiss'):
            vectorstore = FAISS.load_local(args.index_directory, embeddings, args.personality)
        else:
            vectorstore = FAISS(
                embeddings.embed_query, 
                faiss.IndexFlatL2(384), # for openAi embed is 1536
                InMemoryDocstore({}),
                {} #index_to_docstore_id
            )

        retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            human_prefix=personality['user'],
            ai_prefix=personality['llm'],
        )

        prompt = PromptTemplate(input_variables=["history", "input"], template=personality['chat'])
        chain = ConversationChain(
            prompt=prompt,
            llm=llm,
            memory=memory,
            verbose=False,
            #memory=ConversationBufferMemory(
            #    human_prefix=personality['user'],
            #    ai_prefix=personality['llm'],
            #),
        )

        print( f"type {personality['bye']} or /quit to exit" );
        print( chain.predict(input=personality['hi']) )

        while True:
            loser_says_what = input(f"----\n{personality['user']}>> " )
            response = chain.predict(input=loser_says_what)
            print( f"\n{personality['llm']}>> {response}" )

            # idk if this helps...
            #memory.save_context({"input":loser_says_what},{"output":response})
            memory.save_context( {personality['user']:loser_says_what}, {personality['llm']:response}) 

            if "/quit" == loser_says_what or loser_says_what.lower() == personality['bye'].lower():
                break

        print('saving conversation')
        vectorstore.save_local(args.index_directory, args.personality)
        print('saved conversation')


    def createLLM(self, args):
        #callback_manager     Optional[BaseCallbackManager] = None
        #callbacks            Callbacks = None
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
        n_batch      = 512 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_ctx        = 512 # tweak this too...

        # smallest: llama-2-7b-chat.ggmlv3.q2_K.bin   5g
        # biggest:  llama-2-7b-chat.ggmlv3.q8_0.bin  10g
        model_path   = "models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin"

        MuscleLLM.LOG.info( f'loading ${model_path}' )
        print( f'MM: loading ${model_path}' )
        llm = LlamaCpp(
            model_path=args.model_path,
            cache=args.cache,
            echo=args.echo,
            f16_kv=args.f16_kv,
            last_n_tokens_size=args.last_n_tokens_size,
            logits_all=args.logits_all,
            logprobs=args.logprobs,
            lora_base=args.lora_base,
            lora_path=args.lora_path,
            max_tokens=args.max_tokens,
            n_batch=args.n_batch,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_parts=args.n_parts,
            n_threads=args.n_threads,
            repeat_penalty=args.repeat_penalty,
            rope_freq_base=args.rope_freq_base,
            rope_freq_scale=args.rope_freq_scale,
            seed=args.seed,
            #stop=args.stop.split(","),
            streaming=args.streaming,
            suffix=args.suffix,
            #tags=args.tags.split(","),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_mlock=args.use_mlock,
            use_mmap=args.use_mmap,
            verbose=args.verbose,
            vocab_only=args.vocab_only,
            #model_kwargs={ "color":True },
        )

        MuscleLLM.LOG.info( f'loaded ${model_path}' )
        print( f'MMloaded ${model_path}' )
        return llm


if __name__ == "__main__":
    MuscleLLM().main()

# EOF
#############################################################################
