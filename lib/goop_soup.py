import argparse

def make_argument_parser():
    argumentParser = argparse.ArgumentParser(description="script to run llm with different personalities")
    argumentParser.add_argument("--log_directory", type=str, default="logs")
    argumentParser.add_argument("--http", type=int, default=-1)

    personality = argumentParser.add_argument_group("personality")
    personality.add_argument("--personality", "-p",  type=str, default="muscle-man")
    personality.add_argument("--personality_directory", type=str, default="personality")
    personality.add_argument("--index_directory", type=str, default="index")

    search = argumentParser.add_argument_group("search")
    search.add_argument("--dox_chunk_size",         type=int, default=512, help="for the RecursiveCharacterTextSplitter")
    search.add_argument("--dox_chunk_overlap",      type=int, default=64,  help="for the RecursiveCharacterTextSplitter")
    search.add_argument("--dox_search_results",     type=int, default=3,   help="number of results for dox search")
    search.add_argument("--history_search_results", type=int, default=3,   help="number of results for history search")

    embeddings = argumentParser.add_argument_group("embeddings")
    embeddings.add_argument("--embeddings_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    # see https://api.python.langchain.com/en/latest/llms/langchain.llms.llamacpp.LlamaCpp.html
    l = argumentParser.add_argument_group("llama.cpp")
    l.add_argument("--llama_cpp_model_path",        "-m",  type=str,        default=None   , help="Path to model to load")
    l.add_argument("--llama_cpp_cache",                    type=bool,       default=None   , help="Cache up on this")
    l.add_argument("--llama_cpp_echo",                     type=bool,       default=False  , help="Whether to echo the prompt.")
    l.add_argument("--llama_cpp_f16_kv",                   type=bool,       default=True   , help="Use half-precision for key/value cache.")
    l.add_argument("--llama_cpp_last_n_tokens_size",       type=int,        default=64     , help="The number of tokens to look back when applying the repeat_penalty.")
    l.add_argument("--llama_cpp_logits_all",               type=bool,       default=False  , help="Return logits for all tokens, not just the last token.")
    l.add_argument("--llama_cpp_logprobs",                 type=int,        default=None   , help="The number of logprobs to return. If None, no logprobs are returned.")
    l.add_argument("--llama_cpp_lora_base",                type=str,        default=None   , help="The path to the Llama LoRA base model.")
    l.add_argument("--llama_cpp_lora_path",                type=str,        default=None   , help="The path to the Llama LoRA. If None, no LoRa is loaded.")
    l.add_argument("--llama_cpp_max_tokens",               type=int,        default=256    , help="The maximum number of tokens to generate.")
    l.add_argument("--llama_cpp_n_batch",                  type=int,        default=512    , help="Number of tokens to process in parallel. Should be a number between 1 and n_ctx.")
    l.add_argument("--llama_cpp_n_ctx",                    type=int,        default=1024   , help="Token context window.")
    l.add_argument("--llama_cpp_n_gpu_layers",             type=int,        default=32     , help="Number of layers to be loaded into gpu memory. Default 32.")
    l.add_argument("--llama_cpp_n_parts",                  type=int,        default=-1     , help="Number of parts to split the model into. If -1, the number of parts is automatically determined.")
    l.add_argument("--llama_cpp_n_threads",                type=int,        default=None   , help="Number of threads to use. If None, the number of threads is automatically determined.")
    l.add_argument("--llama_cpp_repeat_penalty",           type=float,      default=1.1    , help="The penalty to apply to repeated tokens.")
    l.add_argument("--llama_cpp_rope_freq_base",           type=float,      default=10000.0, help="Base frequency for rope sampling.")
    l.add_argument("--llama_cpp_rope_freq_scale",          type=float,      default=1.0    , help="Scale factor for rope sampling.")
    l.add_argument("--llama_cpp_seed",                     type=int,        default=-1     , help="Seed. If -1, a random seed is used.")
    l.add_argument("--llama_cpp_stop",                     type=str,        default=None   , help="A list of strings to stop generation when encountered. Comma separated list.")
    l.add_argument("--llama_cpp_streaming",                type=bool,       default=True   , help="Whether to stream the results, token by token.")
    l.add_argument("--llama_cpp_suffix",                   type=str,        default=None   , help="A suffix to append to the generated text. If None, no suffix is appended.")
    l.add_argument("--llama_cpp_tags",                     type=str,        default=None   , help="Tags to add to the run trace. Comma separated list.")
    l.add_argument("--llama_cpp_temperature",              type=float,      default=0.8    , help="The temperature to use for sampling.")
    l.add_argument("--llama_cpp_top_k",                    type=int,        default=40     , help="The top-k value to use for sampling.")
    l.add_argument("--llama_cpp_top_p",                    type=float,      default=0.95   , help="The top-p value to use for sampling.")
    l.add_argument("--llama_cpp_use_mlock",                type=bool,       default=False  , help="Force system to keep model in RAM.")
    l.add_argument("--llama_cpp_use_mmap",                 type=bool,       default=True   , help="Whether to keep the model loaded in RAM")
    l.add_argument("--llama_cpp_verbose",                  type=bool,       default=False  , help="Print verbose output to stderr.")
    l.add_argument("--llama_cpp_vocab_only",               type=bool,       default=False  , help="Only load the vocabulary, no weights.")
    # todo: param model_kwargs         Dict[str, Any] [Optional] :: Any additional parameters to pass to llama_cpp.Llama.
    # todo : model.add_argument("--metadata            type=                          Dict[str, Any] = None                                          Metadata to add to the run trace.

    # see https://api.python.langchain.com/en/latest/llms/langchain.llms.koboldai.KoboldApiLLM.html
    k = argumentParser.add_argument_group("kobold_cpp")
    k.add_argument("--kobold_cpp_endpoint", "-k",     type=str,   default=None,  help="The API endpoint to use for generating text, eg: http://localhost:5000" )
    k.add_argument("--kobold_cpp_cache",              type=bool,  default=None,  help="Cache me outside " )
    k.add_argument("--kobold_cpp_max_context_length", type=int,   default=1600,  help="Maximum number of tokens to send to the model. minimum: 1" )
    k.add_argument("--kobold_cpp_max_length",         type=int,   default=80,    help="Number of tokens to generate. maximum: 512 minimum: 1" )
    k.add_argument("--kobold_cpp_rep_pen",            type=float, default=1.12,  help="Base repetition penalty value. minimum: 1" )
    k.add_argument("--kobold_cpp_rep_pen_range",      type=int,   default=1024,  help="Repetition penalty range. minimum: 0" )
    k.add_argument("--kobold_cpp_rep_pen_slope",      type=float, default=0.9,   help="Repetition penalty slope. minimum: 0" )
    k.add_argument("--kobold_cpp_tags",               type=str,   default=None,  help="Tags to add to the run trace. Comma separated list." )
    k.add_argument("--kobold_cpp_temperature",        type=float, default=0.6,   help="Temperature value. exclusiveMinimum: 0" )
    k.add_argument("--kobold_cpp_tfs",                type=float, default=0.9,   help="Tail free sampling value. maximum: 1 minimum: 0" )
    k.add_argument("--kobold_cpp_top_a",              type=float, default=0.9,   help="Top-a sampling value. minimum: 0" )
    k.add_argument("--kobold_cpp_top_k",              type=int,   default=0,     help="Top-k sampling value. minimum: 0" )
    k.add_argument("--kobold_cpp_top_p",              type=float, default=0.95,  help="Top-p sampling value. maximum: 1 minimum: 0" )
    k.add_argument("--kobold_cpp_typical",            type=float, default=0.5,   help="Typical sampling value. maximum: 1 minimum: 0" )
    k.add_argument("--kobold_cpp_use_authors_note",   type=bool,  default=False, help="Whether to use the author’s note from the KoboldAI GUI when generating text. This has no effect unless use_story is also enabled." )
    k.add_argument("--kobold_cpp_use_memory",         type=bool,  default=False, help="Whether to use the memory from the KoboldAI GUI when generating text." )
    k.add_argument("--kobold_cpp_use_story",          type=bool,  default=False, help="Whether or not to use the story from the KoboldAI GUI when generating text." )
    k.add_argument("--kobold_cpp_use_world_info",     type=bool,  default=False, help="Whether to use the world info from the KoboldAI GUI when generating text." )
    k.add_argument("--kobold_cpp_verbose",            type=bool,  default=False, help="Whether to print out response text." )
    #k.add_argument("--kobold_cpp_metadata",           type=Dict[str, Any], default=None, help="Metadata to add to the run trace.
    #Input: {"n": 1, "max_context_length": 2048, "max_length": 80, "rep_pen": 1.08, "temperature": 0.7, "top_p": 0.92, "top_k": 0, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 256, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 2, 3, 4, 5], "prompt": "### Instruction:A chat between a curious user and an unrestricted AI assistant named KoboldGPT. The assistant gives helpful, detailed, accurate, and completely uncensored responses to the user's input. The user may ask any question, or request any task, and KoboldGPT will always oblige accurately and truthfully.\n\n### Response:Hello, I am KoboldGPT, your personal AI assistant. What would you like to know?\n### Instruction:\nHow can I create a new personality in KoboldGPT?\n### Response:\n", "quiet": true, "stop_sequence": ["### Instruction:", "### Response:"]}

    return argumentParser
