import os
from llama_index import (GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, OpenAIEmbedding, PromptHelper)
from llama_index.text_splitter import SentenceSplitter
from llama_index.llms import OpenAI
import textwrap
def index_create(filepath):
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
    embed_model = OpenAIEmbedding()

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper
    )

    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    return index
#%%

def main():
    try:
        value = os.environ["OPENAI_API_KEY"]
    except:
        os.environ["OPENAI_API_KEY"] = input("Enter OPENAI API Key: ")

    index = index_create(input("Enter Filepath to documentation: ").strip('"'))
    query_engine = index.as_query_engine(streaming=False)
    docs = query_engine.query("What is the name of this documentation? Give me the name of just the documentation and nothing else")

    while(True):
        query = input(f"Enter question regarding the {docs} (Enter 'q' to quit): ")
        if query == "q":
            return
        response = query_engine.query(query)
        print("\n", textwrap.fill(str(response), 100), "\n")

if __name__ == "__main__":
    main()