"""
This STORM Wiki pipeline powered by Ollama llama3.1:8b and local retrieval model that uses Qdrant.
You need to have Ollama running locally on port 11434 with llama3.1:8b model downloaded.

To download the model in Ollama:
ollama pull llama3.1:8b

You will also need an existing Qdrant vector store either saved in a folder locally offline or in a server online.
If not, then you would need a CSV file with documents, and the script is going to create the vector store for you.
The CSV should be in the following format:
content  | title  |  url  |  description
I am a document. | Document 1 | docu-n-112 | A self-explanatory document.
I am another document. | Document 2 | docu-l-13 | Another self-explanatory document.

Notice that the URL will be a unique identifier for the document so ensure different documents have different urls.

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""

import os
import requests
import json
from argparse import ArgumentParser

from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.rm import VectorRM
from knowledge_storm.lm import OllamaClient
from knowledge_storm.utils import QdrantVectorStoreManager


# class OllamaModel(OllamaClient):
#     """
#     Ollama language model for STORM.
#     """
    
#     def __init__(
#         self,
#         model: str,
#         max_tokens: int = 8192,
#         temperature: float = 1.0,
#         top_p: float = 0.9,
#         base_url: str = "http://localhost:11434"
#     ):
#         self.model = model
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.top_p = top_p
#         self.base_url = base_url
    
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate text using Ollama API."""
#         url = f"{self.base_url}/api/generate"
        
#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "temperature": self.temperature,
#                 "top_p": self.top_p,
#                 "num_predict": self.max_tokens,
#             }
#         }
        
#         try:
#             response = requests.post(url, json=payload, timeout=300)
#             response.raise_for_status()
#             result = response.json()
#             return result.get("response", "")
#         except requests.exceptions.RequestException as e:
#             print(f"Error calling Ollama API: {e}")
#             return ""


def main(args):
    # Initialize the language model configurations
    engine_lm_configs = STORMWikiLMConfigs()
    
    ollama_kwargs = {
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.9,
    }

    # STORM is a LM system so different components can be powered by different models.
    # Here we use llama3.1:8b for all components with maximum tokens for best performance.
    # conv_simulator_lm = OllamaModel(**ollama_kwargs)
    # question_asker_lm = OllamaModel(**ollama_kwargs)
    # outline_gen_lm = OllamaModel(**ollama_kwargs)
    # article_gen_lm = OllamaModel(**ollama_kwargs)
    # article_polish_lm = OllamaModel(**ollama_kwargs)
    
    conv_simulator_lm = OllamaClient(model="llama3.1", port=11434, **ollama_kwargs)
    question_asker_lm = OllamaClient(model="llama3.1", port=11434, **ollama_kwargs)
    outline_gen_lm = OllamaClient(model="llama3.1", port=11434, **ollama_kwargs)
    article_gen_lm = OllamaClient(model="llama3.1", port=11434, **ollama_kwargs)
    article_polish_lm = OllamaClient(model="llama3.1", port=11434, **ollama_kwargs)

    engine_lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    engine_lm_configs.set_question_asker_lm(question_asker_lm)
    engine_lm_configs.set_outline_gen_lm(outline_gen_lm)
    engine_lm_configs.set_article_gen_lm(article_gen_lm)
    engine_lm_configs.set_article_polish_lm(article_polish_lm)

    # Initialize the engine arguments
    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # Create / update the vector store with the documents in the csv file
    if args.csv_file_path:
        kwargs = {
            "file_path": args.csv_file_path,
            "content_column": "content",
            "title_column": "title",
            "url_column": "url",
            "desc_column": "description",
            "batch_size": args.embed_batch_size,
            "vector_db_mode": args.vector_db_mode,
            "collection_name": args.collection_name,
            "embedding_model": args.embedding_model,
            "device": args.device,
        }
        if args.vector_db_mode == "offline":
            QdrantVectorStoreManager.create_or_update_vector_store(
                vector_store_path=args.offline_vector_db_dir, **kwargs
            )
        elif args.vector_db_mode == "online":
            QdrantVectorStoreManager.create_or_update_vector_store(
                url=args.online_vector_db_url,
                api_key=os.getenv("QDRANT_API_KEY"),
                **kwargs
            )

    # Setup VectorRM to retrieve information from your own data
    rm = VectorRM(
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        device=args.device,
        k=engine_args.search_top_k,
    )

    # initialize the vector store, either online (store the db on Qdrant server) or offline (store the db locally):
    if args.vector_db_mode == "offline":
        rm.init_offline_vector_db(vector_store_path=args.offline_vector_db_dir)
    elif args.vector_db_mode == "online":
        rm.init_online_vector_db(
            url=args.online_vector_db_url, api_key=os.getenv("QDRANT_API_KEY")
        )

    # Initialize the STORM Wiki Runner
    runner = STORMWikiRunner(engine_args, engine_lm_configs, rm)

    # run the pipeline
    topic = input("Topic: ")
    runner.run(
        topic=topic,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
    )
    runner.post_run()
    runner.summary()


if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/ollama_retrieval",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="Maximum number of threads to use. The information seeking part and the article generation"
        "part can speed up by using multiple threads. Consider reducing it if keep getting "
        'errors when calling Ollama API.',
    )
    # provide local corpus and set up vector db
    parser.add_argument(
        "--collection-name",
        type=str,
        default="my_documents",
        help="The collection name for vector store.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
        help="The embedding model for vector store.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device used to run the retrieval model (cpu, cuda, mps, etc).",
    )
    parser.add_argument(
        "--vector-db-mode",
        type=str,
        choices=["offline", "online"],
        help="The mode of the Qdrant vector store (offline or online).",
    )
    parser.add_argument(
        "--offline-vector-db-dir",
        type=str,
        default="./vector_store",
        help="If use offline mode, please provide the directory to store the vector store.",
    )
    parser.add_argument(
        "--online-vector-db-url",
        type=str,
        help="If use online mode, please provide the url of the Qdrant server.",
    )
    parser.add_argument(
        "--csv-file-path",
        type=str,
        default=None,
        help="The path of the custom document corpus in CSV format. The CSV file should include "
        "content, title, url, and description columns.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding the documents in the csv file.",
    )
    # stage of the pipeline
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="If True, simulate conversation to research the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If True, polish the article by adding a summarization section and (optionally) removing "
        "duplicate content.",
    )
    # hyperparameters for the pre-writing stage
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="Maximum number of questions in conversational question asking.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Maximum number of perspectives to consider in perspective-guided question asking.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="Top k search results to consider for each search query.",
    )
    # hyperparameters for the writing stage
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=3,
        help="Top k collected references for each section title.",
    )
    parser.add_argument(
        "--remove-duplicate",
        action="store_true",
        help="If True, remove duplicate content from the article.",
    )
    main(parser.parse_args())