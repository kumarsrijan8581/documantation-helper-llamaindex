from dotenv import load_dotenv
import os
from unstructured.partition.auto import partition
from llama_index.readers.file import UnstructuredReader
import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext ,StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import service_context,download_loader
from llama_index.core.storage import storage_context
from llama_index.vector_stores.pinecone import PineconeVectorStore

from pinecone import Pinecone


load_dotenv()
# pinecone.Pinecone(
#    api_key=os.getenv("PINECONE_API_KEY"),
#    environment=os.getenv("PINECONE_ENVIRONMENT"),
# )
if __name__ == "__main__":
    print("going to ingest pinecone documentation...")
    dir_reader = SimpleDirectoryReader(input_dir="./llamaindex-docs", file_extractor={".html": UnstructuredReader()})
    documents = dir_reader.load_data()
    node_parser=SimpleNodeParser.from_defaults(chunk_size=500,chunk_overlap=20)
    llm=OpenAI(model="gpt-3.5-turbo",temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
    # service_context=ServiceContext.from_defaults(llm=llm, embed_model=embed_model,node_parser=node_parser)

    index_name = "llamaindex-documentation-helper"
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("finished ingestion...")
