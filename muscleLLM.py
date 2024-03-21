#!/usr/bin/env python
#############################################################################

import argparse
import json
import logging
import os
import re

from langchain.llms import (
        LlamaCpp,
        KoboldApiLLM
)

from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate, 
    SystemMessage, 
    SystemMessagePromptTemplate
)

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
import numpy as np
from tabulate import tabulate

from lib.utils            import read_txt_files, default_dict
from lib.goop_soup        import make_argument_parser
from lib.document_loaders import load_document

# unfortunately, flask is frakncnasdf!@#$1!!
#from flask import Flask, request, jsonify

from http.server import BaseHTTPRequestHandler, HTTPServer

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
        self.argumentParser = make_argument_parser()

    def main(self):
        args = self.argumentParser.parse_args()

        personality = self.readPersonality( args )

        embeddings, history, dox, memory = self.youreTalkingAboutMemory(args,personality)

        prompt, input_variables = self.createPrompt( personality )
        print( 'prompt is', prompt )

        llm = self.createLLM(args)

        chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

        # TODO: clean these up
        help, commands = self.getHelp(args, personality, embeddings, history, dox, memory, prompt, input_variables, llm, chain)

        if -1 == args.http:
            self.shell(args, personality, embeddings, history, dox, memory, prompt, input_variables, llm, chain, help, commands)
        else:
            self.http(args, personality, embeddings, history, dox, memory, prompt, input_variables, llm, chain, help, commands)


    def shell(self, args, personality, embeddings, history, dox, memory, prompt, input_variables, llm, chain, help, commands):
        commands["/help"]({})
        #print( chain.predict(**input_variables) )
        
        #response = llm(prompt.format_messages(**input_variables))
        #response = chain(input_variables)
        #print( response )

        print("P1: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(personality['hi'])
        print("P1: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        llm = personality['llm']
        user = personality['user']

        while True:
            loser_says_what = input(f"----\n{user}>> " )
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
                print("P1: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(f"unknown command:{command}")
                continue

            # prepare the inputs

            input_variables['input'] = loser_says_what

            #print("HISTORY SEARCH")
            input_variables['history'], matched_docs = self.searchVectorStore(
                loser_says_what, 
                history,
                embeddings,
            )

            if 'dox' in input_variables:
                #print("DOX SEARCH")
                input_variables['dox'], matched_docs = self.searchVectorStore(
                    loser_says_what, 
                    dox,
                    embeddings,
                )


            # valerie: disable this for now
            input_variables['history'] = ''
            input_variables['dox'] = ''


            # predict, display, and save the results

            response = chain.predict(**input_variables,
                human_prefix="Mordecai",
                ai_prefix="Muscle Man",
            )

            # valerie: idk what this was supposed to accomplish...
            # for message in self.splitChatMessages(response, personality, False, not False ):
            #     print("P2: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #     print("suck", message )
            #     print("P2: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # continue


            #response = re.sub(r'^AI:\s*',    f'{llm}:',  response, flags=re.MULTILINE)
            #response = re.sub(r'^Human:\s*', f'{user}:', response, flags=re.MULTILINE)
            #response = response.strip()
            #response = chain(input_variables)
            #response = llm(prompt.format_messages(**input_variables))
            print("P3: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print( f"\n{llm}>> {response}" )
            print("P3: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            if not response:
                print(f"\n{llm}>> *farts*")

            # save_context(inputs: Dict[str, Any], outputs: Dict[str, str]) → None
            memory.save_context( 
                {user:loser_says_what}, 
                {llm:response},
                #{"input":loser_says_what}, 
                #{"output":response},
            ) 

            # handle bye 

            if loser_says_what.lower() == personality['bye'].lower():
                break

        commands["/save"]({})




    def http(self, args, personality, embeddings, history, dox, memory, prompt, input_variables, llm, chain, help, commands):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()

                message = "Hey... sry.. try post instead, yo..."
                self.wfile.write(bytes(message, "utf8"))

            def do_POST(self):
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()

                content_len = int(self.headers.get('Content-Length'))
                request = self.rfile.read(content_len)

                input_variables['input'] = request
                input_variables['history'] = ''
                input_variables['dox'] = ''
                response = chain.predict(**input_variables
                    #, human_prefix="Mordecai"
                    #, ai_prefix="Muscle Man"
                )
                message = response #"Hello, World! Here is a POST response"
                self.wfile.write(bytes(message, "utf8"))

        with HTTPServer(('', args.http), Handler) as server:
            server.serve_forever()

        #app = Flask('bcrack')
        #self.chain = chain
        #
        #@app.route('/', methods=['POST'])
        #def flasko():
        #    input_variables['input'] = request.data
        #    input_variables['history'] = ''
        #    input_variables['dox'] = ''
        #    response = self.chain.predict(**input_variables
        #        #, human_prefix="Mordecai"
        #        #, ai_prefix="Muscle Man"
        #    )
        #    return response
        #
        #app.run(debug=True, host='localhost', port=args.http)
        #

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
        want=3
        search=10
        minRating=.2
        debug=True
        commands = {
            "/quit"     : lambda values: True,
            "/help"     : lambda values: self.printTable(help,{"name": "input", "value": "result"}),
            "/dox"      : lambda values: (self.searchVectorStore(values, dox,     embeddings, want, search, minRating, debug ))[1],
            "/history"  : lambda values: (self.searchVectorStore(values, history, embeddings, want, search, minRating, debug ))[1],
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
            embeddings_size = len(embeddings.embed_query("you know who else likes embeddings? my mom!"))
            vectorstore = FAISS(
                embeddings.embed_query, 
                faiss.IndexFlatL2(embeddings_size),
                InMemoryDocstore({}),
                {} #index_to_docstore_id
            )
        return vectorstore


    def createPrompt(self, personality, debug=False):
        input_variables = {}
        for k in re.findall(r'\{([a-zA-Z]+)\}', personality['chat']):
            input_variables[k] = ""
        chat = personality["chat"]
        return ChatPromptTemplate.from_messages(self.splitChatMessages(chat, personality, True, debug)), input_variables


    def createPromptOld(self, personality):
        # old mechanism...
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


    def splitChatMessages(self, chat, personality, useTemplates=True, debug=True):
        debug = True

        messages = []
        if debug:
            print(f"--------------------------------------------------------------\n{chat}\n-----------------------------------")

        who = "system"
        lines = []
        speakerx = r'^({[a-z]+}|[a-z ]+):\s*'
        lineNumber = 0

        starting = True


        for line in chat.split("\n"):
            lineNumber = 1 + lineNumber
            if starting and "" == line.strip():
                continue
            starting = False
                
            if debug:
                print( f"{lineNumber:2d} {line}" )
            matchSpeaker = re.match(speakerx, line, re.IGNORECASE)
            if matchSpeaker:
                newWho = matchSpeaker.group(1)
                if not useTemplates:
                    macro = re.match(r'^{([a-z]+)}', newWho, re.IGNORECASE)
                    if macro:
                       macro = macro.group(1)
                       if not macro in personality:
                           raise ValueError( f'unknown macro {macro} on line {lineNumber}' )
                       newWho = personality[ macro ]
                    else:
                       print(f"fucking you, {newWho}")
                if not newWho == who:
                    self.saveMessage(personality, who, lines, messages, useTemplates, debug)
                    who = newWho
                    lines = []
                line = re.sub(speakerx, '', line, flags=re.IGNORECASE).strip()
                if not line:
                    continue
            if debug:
                print( f"{who} -> {line}") 
            lines.append( line )
        self.saveMessage(personality, who, lines, messages,useTemplates, debug)

        if debug:
            print( "messages:", messages )

        return messages


    def saveMessage(self, personality, who, lines, messages, useTemplates, debug=False):
        if 0 == len(lines):
            return
        txt = "\n".join(lines)
        if debug:
            print(f"save {who} {len(lines)} > {txt}")
        if not useTemplates:
            return messages.append({who:txt})
        w = who.lower()
        if w in set('ai {llm}'.split()):
            return messages.append(AIMessagePromptTemplate.from_template(txt))
        if w in set('sys system inst'.split()):
            return messages.append(SystemMessagePromptTemplate.from_template(txt))
        return messages.append(HumanMessagePromptTemplate.from_template(txt))


    def createLLM(self, args):
        if args.llama_cpp_model_path:
            return self.createLlamaCpp(args)
        if args.kobold_cpp_endpoint:
            return self.createKoboldCpp(args)
        raise ValueError("need to define either llama_cpp_model_path or kobold_cpp_endpoint")

    def createLlamaCpp(self,args):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        MuscleLLM.LOG.info( f'loading {args.llama_cpp_model_path}' )
        print( f'MM: llama.cpp: loading {args.llama_cpp_model_path}' )
        llm = LlamaCpp(
            model_path=args.llama_cpp_model_path,
            cache=args.llama_cpp_cache,
            echo=args.llama_cpp_echo,
            f16_kv=args.llama_cpp_f16_kv,
            last_n_tokens_size=args.llama_cpp_last_n_tokens_size,
            logits_all=args.llama_cpp_logits_all,
            logprobs=args.llama_cpp_logprobs,
            lora_base=args.llama_cpp_lora_base,
            lora_path=args.llama_cpp_lora_path,
            max_tokens=args.llama_cpp_max_tokens,
            n_batch=args.llama_cpp_n_batch,
            n_ctx=args.llama_cpp_n_ctx,
            n_gpu_layers=args.llama_cpp_n_gpu_layers,
            n_parts=args.llama_cpp_n_parts,
            n_threads=args.llama_cpp_n_threads,
            repeat_penalty=args.llama_cpp_repeat_penalty,
            rope_freq_base=args.llama_cpp_rope_freq_base,
            rope_freq_scale=args.llama_cpp_rope_freq_scale,
            seed=args.llama_cpp_seed,
            #stop=args.llama_cpp_stop.split(","),
            streaming=args.llama_cpp_streaming,
            suffix=args.llama_cpp_suffix,
            #tags=args.llama_cpp_tags.split(","),
            temperature=args.llama_cpp_temperature,
            top_k=args.llama_cpp_top_k,
            top_p=args.llama_cpp_top_p,
            use_mlock=args.llama_cpp_use_mlock,
            use_mmap=args.llama_cpp_use_mmap,
            verbose=args.llama_cpp_verbose,
            vocab_only=args.llama_cpp_vocab_only,
            #model_kwargs.llama_cpp_{ "color":True },
        )

        MuscleLLM.LOG.info( f'loaded {args.llama_cpp_model_path}' )
        print( f'MM: kobold.cpp: loaded {args.llama_cpp_model_path}' )
        return llm

    def createKoboldCpp(self,args):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        MuscleLLM.LOG.info( f'loading {args.kobold_cpp_endpoint}' )
        print( f'MM: kobold.cpp: loading {args.kobold_cpp_endpoint}' )

        llm = KoboldApiLLM(endpoint="http://192.168.1.144:5000", max_length=80)
        llm = KoboldApiLLM(
            endpoint=args.kobold_cpp_endpoint,
            cache=args.kobold_cpp_cache,
            max_context_length=args.kobold_cpp_max_context_length,
            max_length=args.kobold_cpp_max_length,
            rep_pen=args.kobold_cpp_rep_pen,
            rep_pen_range=args.kobold_cpp_rep_pen_range,
            rep_pen_slope=args.kobold_cpp_rep_pen_slope,
            tags=args.kobold_cpp_tags,
            temperature=args.kobold_cpp_temperature,
            tfs=args.kobold_cpp_tfs,
            top_a=args.kobold_cpp_top_a,
            top_k=args.kobold_cpp_top_k,
            top_p=args.kobold_cpp_top_p,
            typical=args.kobold_cpp_typical,
            use_authors_note=args.kobold_cpp_use_authors_note,
            use_memory=args.kobold_cpp_use_memory,
            use_story=args.kobold_cpp_use_story,
            use_world_info=args.kobold_cpp_use_world_info,
            verbose=args.kobold_cpp_verbose,
        )

        MuscleLLM.LOG.info( f'loaded {args.kobold_cpp_endpoint}' )
        print( f'MMloaded {args.kobold_cpp_endpoint}' )
        return llm


    # return value from 1 to 10-ish
    def relevance(self, query, txt, embeddings, minRating=.2):
        q = embeddings.embed_query(query)
        t = embeddings.embed_query(txt)
        dot = np.dot(q,t)
        if dot < minRating:
            return 1, dot
        return 1 + (dot - minRating) * 44, dot


    def relevanceSearchVectorStore(self, query, vectorstore, embedding, want=3, search=10, minRating=.2, debug=False ):
        results = []
        matched_docs = vectorstore.similarity_search(query, search)
        seen = {}

        for doc in matched_docs:
            txt = doc.page_content
            if txt in seen:
                continue
            seen[ txt ] = True
            rating, dot = self.relevance(query, doc.page_content, embedding, minRating)
            if rating >= minRating:
                results.append([txt,rating,dot])

        results = sorted(results, key=lambda x: x[1], reverse=True)[:want]
        if debug:
            l77 = "-"*77
            print(f"{l77}\nTop {len(results)} Results for '{query}'")
            n = 0
            for result in results:
                n = n + 1
                txt, rating, dot = results[0]
                print(f'#{n:2d} {rating:3.1f} {dot:2f} > {txt}')
                print("-"*44)
            print(f"{l77}\n\n")
        return results


    def searchVectorStore(self, query, vectorstore, embedding, want=3, search=10, minRating=.2, debug=False ):
        if isinstance(query, list):
            query = " ".join(query)
        elif not isinstance(query, str):
            raise ValueError("query should be a string or a list")
        results = self.relevanceSearchVectorStore(query, vectorstore, embedding, want, search, minRating, debug )
        txt = ""
        for result in results:
            txt = txt + result[0] + "\n"
        return txt, False


    def printTable(self, uDict, labels = {"name": "name", "value": "value"} ):
        output_list = [{labels["name"]: key, labels["value"]: value} for key, value in uDict.items()]
        print(tabulate(pd.DataFrame( output_list ), headers='keys', tablefmt='psql', showindex=False))


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


if __name__ == "__main__":
    MuscleLLM().main()

# EOF
#############################################################################
