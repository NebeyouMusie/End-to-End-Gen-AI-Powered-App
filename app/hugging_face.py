from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from config import load_config, get_huggingface_token

# load app configuration
load_config()

# Hugging Face LLM
def huggingface_llm():
    # setup LLM from Hugging Face
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        model_kwargs = {'max_length': 128},
        temperature=0.1,
        huggingfacehub_api_token=get_huggingface_token(),
    )
    return llm

# Hugging Face Embedding
def huggingface_embedding():
    # setup embedding
    embeddings = HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',  #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
    )

    return embeddings

 