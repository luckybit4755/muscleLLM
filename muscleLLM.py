#!/usr/bin/env python
#############################################################################

import argparse
import logging
import os
import re

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import (
    ConversationChain,
    ConversationalRetrievalChain,
)
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

import pandas as pd
from tabulate import tabulate

from utils import read_txt_files, default_dict, load_document

#############################################################################

#############################################################################
#
#
#
#
#
#############################################################################
class MuscleLLM:
    LOG = logging.getLogger(__name__)
    DEFAULT_PERSONALITY_OPTIONS = { "bye": "bye", "hi": "hi", "llm": "AI", "user": "Human"}

    def __init__(self):
        self.argumentParser = self.makeArgumentParser()

    def main(self):
        
        args = self.argumentParser.parse_args()

        personality = self.readPersonality( args )

        embeddings, history, dox, memory = self.youreTalkingAboutMemory(args,personality)

        prompt, input_variables = self.createPrompt( personality )

        llm = self.createLLM(args)

        chain = LLMChain(prompt=prompt, llm=llm)

        # TODO: clean this up
        help, commands = self.getHelp( args, personality, embeddings, history, dox, memory, prompt, input_variables, llm, chain )

        ########################################################################
        # weaksauce shell

        commands["/help"]({})
        print( chain.predict(**input_variables) )

        while True:
            loser_says_what = input(f"----\n{personality['user']}>> " )
            input_parts = loser_says_what.strip().split()
            if 0 == len( input_parts ):
                continue

            # handle commands 
            command = input_parts[0]
            if command in commands:
                shouldQuit = commands[command](input_parts[1:])
                if shouldQuit:
                    break
                continue
            if "/" == command[0]:
                print(f"unknown command:{command}")
                continue

            # prepare the inputs

            input_variables['input'] = loser_says_what

            input_variables['history'], matched_docs = self.searchVectorStore(
                loser_says_what, 
                history,
                args.history_search_results,
            )

            if 'dox' in input_variables:
                input_variables['dox'], matched_docs = self.searchVectorStore(
                    loser_says_what, 
                    dox,
                    args.dox_search_results,
                )

            # predict, display, and save the results

            response = chain.predict(**input_variables)
            print( f"\n{personality['llm']}>> {response}" )
            memory.save_context( {personality['user']:loser_says_what}, {personality['llm']:response}) 

            # handle bye 

            if loser_says_what.lower() == personality['bye'].lower():
                break

        commands["/save"]({})


    def readPersonality(self, args):    
        return default_dict( 
            read_txt_files( f'{args.personality_directory}/{args.personality}' ),
            MuscleLLM.DEFAULT_PERSONALITY_OPTIONS
        )

    def getHelp(self, args, personality, embeddings, history, dox, memory, prompt, input_variables, llm, chain ):
        help = {
            personality['bye']: "say bye and quit the chat",
            "/quit"           : "don't say bye, just quit",
            "/help"           : "print this thing",
            "/dox"            : "search and print dox matching the text",
            "/history"        : "search and print history matching the text",
            "/add <filename>" : "add a file to the dox",
            "/save"           : "save history and dox",
        }
        commands = {
            "/quit"     : lambda values: True,
            "/help"     : lambda values: self.printTable(help,{"name": "input", "value": "result"}),
            "/dox"      : lambda values: self.searchDox(values, dox, args.dox_search_results),
            "/history"  : lambda values: self.searchDox(values, history, args.history_search_results),
            "/add"      : lambda values: self.addDox(values, dox, args),
            "/save"     : lambda values: self.saveDox(history, dox, args.index_directory, args.personality)
        }
        return help, commands


    def youreTalkingAboutMemory(self, args, personality):
        embeddings = HuggingFaceEmbeddings(model_name=args.embeddings_model_name)
        vectorstore = self.createVectorStore(args.index_directory, embeddings, args.personality)
        dox = self.createVectorStore(args, embeddings, f'{args.personality}-dox')

        retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            human_prefix=personality['user'],
            ai_prefix=personality['llm'],
        )

        return embeddings, vectorstore, dox, memory


    def createVectorStore(self, index_directory, embeddings, storeName):
        if os.path.exists(f'{index_directory}/{storeName}.faiss'):
            vectorstore = FAISS.load_local(index_directory, embeddings, storeName)
        else:
            vectorstore = FAISS(
                embeddings.embed_query, 
                faiss.IndexFlatL2(384), # FIXME: let's not hard code stuff.. for openAi embed is 1536
                InMemoryDocstore({}),
                {} #index_to_docstore_id
            )
        return vectorstore


    def createPrompt(self, personality):
        template = personality['chat']
        for k,v in personality.items():
            template = template.replace("{" + k + "}",v)

        pattern = r'\{([a-zA-Z]+)\}'  # Matches {xxx} and captures xxx
        variables = re.findall(pattern, template)

        input_variables = {}
        for v in variables:
            input_variables[v] = ""

        prompt = PromptTemplate( input_variables=variables, template=template,)
        return prompt, input_variables


    def createLLM(self, args):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        MuscleLLM.LOG.info( f'loading ${args.model_path}' )
        print( f'MM: loading ${args.model_path}' )
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

        MuscleLLM.LOG.info( f'loaded ${args.model_path}' )
        print( f'MMloaded ${args.model_path}' )
        return llm


    def searchVectorStore(self, query, vectorstore, maxDox=3):
        MuscleLLM.LOG.info( f'finding documents for {query}' )
        matched_docs = vectorstore.similarity_search(query, maxDox)
        txt = ""
        for doc in matched_docs:
            txt = txt + doc.page_content + " \n\n "
        MuscleLLM.LOG.info( f'found documents for {query}: {matched_docs}' )
        return txt, matched_docs


    def printTable(self, uDict, labels = {"name": "name", "value": "value"} ):
        output_list = [{labels["name"]: key, labels["value"]: value} for key, value in uDict.items()]
        print(tabulate(pd.DataFrame( output_list ), headers='keys', tablefmt='psql', showindex=False))

    
    def searchDox(self, query, vectorstore, maxDox=3):
        query = " ".join( query )
        matched_docs = vectorstore.similarity_search(query, maxDox)
        print("-"*77)
        print("search for:", query)
        for doc in matched_docs:
            print(">>>", doc, doc.page_content )
            print("-"*33)
        print("-"*77)


    # TODO: scrounge https://www.mlq.ai/autogpt-langchain-research-assistant/
    def addDox(search, filenames, vectorstore, args):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.dox_chunk_size, chunk_overlap=args.dox_chunk_overlap)
        for filename in filenames:
            print(f"loading document", filename)
            document = load_document( filename )
            if document:
                print(f"loaded document", filename, type(document) )
                texts = text_splitter.split_documents(document)
                vectorstore.add_documents(texts)
                print(f"added document", filename, type(texts) )
            else:
                print(f"failed to load document", filename)


    # TODO: track dirty
    def saveDox(self, history, dox, index_directory, personality ):
        print('saving conversation')
        history.save_local(index_directory, personality)
        print('saved conversation')
        print('saving dox')
        dox.save_local(index_directory, personality+"-dox")
        print('saved dox')


    def makeArgumentParser(self):
        argumentParser = argparse.ArgumentParser(description="script to run llm with different personalities")
        argumentParser.add_argument("--log_directory", type=str, default="logs")

        personality = argumentParser.add_argument_group("personality")
        personality.add_argument("--personality", "-p",  type=str, default="muscle-man")
        personality.add_argument("--personality_directory", type=str, default="personality")
        personality.add_argument("--index_directory", type=str, default="index")

        search = argumentParser.add_argument_group("search")
        search.add_argument("--dox_chunk_size",         type=int, default=512, help="for the RecursiveCharacterTextSplitter")
        search.add_argument("--dox_chunk_overlap",      type=int, default=64,  help="for the RecursiveCharacterTextSplitter")
        search.add_argument("--dox_search_results",     type=int, default=3,   help="number of results for dox search")
        search.add_argument("--history_search_results", type=int, default=3,   help="number of results for history search")

        model = argumentParser.add_argument_group("model")
        model.add_argument("--model_path",        "-m",  type=str,        required=True  , help="Path to model to load")
        model.add_argument("--cache",                    type=bool,       default=None)
        model.add_argument("--echo",                     type=bool,       default=False  , help="Whether to echo the prompt.")
        model.add_argument("--f16_kv",                   type=bool,       default=True   , help="Use half-precision for key/value cache.")
        model.add_argument("--last_n_tokens_size",       type=int,        default=64     , help="The number of tokens to look back when applying the repeat_penalty.")
        model.add_argument("--logits_all",               type=bool,       default=False  , help="Return logits for all tokens, not just the last token.")
        model.add_argument("--logprobs",                 type=int,        default=None   , help="The number of logprobs to return. If None, no logprobs are returned.")
        model.add_argument("--lora_base",                type=str,        default=None   , help="The path to the Llama LoRA base model.")
        model.add_argument("--lora_path",                type=str,        default=None   , help="The path to the Llama LoRA. If None, no LoRa is loaded.")
        model.add_argument("--max_tokens",               type=int,        default=256    , help="The maximum number of tokens to generate.")
        model.add_argument("--n_batch",                  type=int,        default=512    , help="Number of tokens to process in parallel. Should be a number between 1 and n_ctx.")
        model.add_argument("--n_ctx",                    type=int,        default=1024   , help="Token context window.")
        model.add_argument("--n_gpu_layers",             type=int,        default=32     , help="Number of layers to be loaded into gpu memory. Default 32.")
        model.add_argument("--n_parts",                  type=int,        default=-1     , help="Number of parts to split the model into. If -1, the number of parts is automatically determined.")
        model.add_argument("--n_threads",                type=int,        default=None   , help="Number of threads to use. If None, the number of threads is automatically determined.")
        model.add_argument("--repeat_penalty",           type=float,      default=1.1    , help="The penalty to apply to repeated tokens.")
        model.add_argument("--rope_freq_base",           type=float,      default=10000.0, help="Base frequency for rope sampling.")
        model.add_argument("--rope_freq_scale",          type=float,      default=1.0    , help="Scale factor for rope sampling.")
        model.add_argument("--seed",                     type=int,        default=-1     , help="Seed. If -1, a random seed is used.")
        model.add_argument("--stop",                     type=str,        default=None   , help="A list of strings to stop generation when encountered. Comma separated list.")
        model.add_argument("--streaming",                type=bool,       default=True   , help="Whether to stream the results, token by token.")
        model.add_argument("--suffix",                   type=str,        default=None   , help="A suffix to append to the generated text. If None, no suffix is appended.")
        model.add_argument("--tags",                     type=str,        default=None   , help="Tags to add to the run trace. Comma separated list.")
        model.add_argument("--temperature",              type=float,      default=0.8    , help="The temperature to use for sampling.")
        model.add_argument("--top_k",                    type=int,        default=40     , help="The top-k value to use for sampling.")
        model.add_argument("--top_p",                    type=float,      default=0.95   , help="The top-p value to use for sampling.")
        model.add_argument("--use_mlock",                type=bool,       default=False  , help="Force system to keep model in RAM.")
        model.add_argument("--use_mmap",                 type=bool,       default=True   , help="Whether to keep the model loaded in RAM")
        model.add_argument("--verbose",                  type=bool,       default=False  , help="Print verbose output to stderr.")
        model.add_argument("--vocab_only",               type=bool,       default=False  , help="Only load the vocabulary, no weights.")
        # todo: param model_kwargs         Dict[str, Any] [Optional] :: Any additional parameters to pass to llama_cpp.Llama.
        # todo : model.add_argument("--metadata            type=                          Dict[str, Any] = None                                          Metadata to add to the run trace.
        model.add_argument("--embeddings_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

        return argumentParser

if __name__ == "__main__":
    MuscleLLM().main()

# EOF
#############################################################################
