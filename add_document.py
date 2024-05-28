import logging
import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


api_key=os.environ.get("PINECONE_API_KEY")


encironment = os.environ.get("PINECONE_ENV")


def initialize_vectorstore():
    index_name = os.environ.get("PINECONE_INDEX")
    embeddings = OpenAIEmbeddings()
    api_key = os.environ.get("PINECONE_API_KEY")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=api_key,
    )

    return vectorstore

if __name__ == "__main__":
    file_path = sys.argv[1]
    loader = UnstructuredPDFLoader(file_path)
    raw_docs = loader.load()
    logger.info("Loaded %d documents", len(raw_docs))

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_docs)
    logger.info("Split %d documents", len(docs))

    vectorstore = initialize_vectorstore()

    vectorstore.add_documents(docs)