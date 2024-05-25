import os
from dotenv import load_dotenv

def load_config():
    return load_dotenv()

def get_huggingface_token():
    return os.getenv('HUGGINGFACEHUB_API_TOKEN')

