# End to End Gen AI Powered App
 - In this project I have built an end to end langchain project using hugging open source llm models such as Mistral and also open source embedding models. 

![Streamlit Web App Interface](./images/app%20interface.png)

## DEMO
 - You can check the project live [here](https://8511-01hwj8ynshjz7spkr595x77ec2.cloudspaces.litng.ai/)

## Description
- This project showcase the implementation of an advanced RAG system that uses Hugging Face as an llm to retrieve information from different PDF documents.

Steps I followed:
1. I have used the `PyPdfDirectoryLoader` from the `langchain_community` document loader to load the PDF documents from the `us-census-data` directory.
2. transformed each text into a chunk of `1000` using the `RecursiveCharacterTextSplitter` imported from the `langchain.text_splitter`
3. stored the vector embeddings which were made using the `HuggingFaceBgeEmbeddings` using the `FAISS` vector store.
4. setup the llm `HuggingFaceEndpoint` with the model name `mistralai/Mistral-7B-Instruct-v0.2`
5. Setup `PromptTemplate`
6. Setup `vector_embedding` function to enbedd the documents and store them in the `FAISS` vectorstore
7. finally created the `RetrievalQA` for chaining `llm`, `prompt` and `retriever`.

## Libraries Used
 - langchain==0.1.20
 - langchain-community==0.0.38
 - langchain-huggingface==0.0.1
 - faiss-cpu==1.8.0
 - python-dotenv==1.0.1

## Installation
 1. Prerequisites
    - Git
    - Command line familiarity
 2. Clone the Repository: `git clone https://github.com/NebeyouMusie/End-to-End-Gen-AI-Powered-App.git`
 3. Create and Activate Virtual Environment (Recommended)
    - `python -m venv venv`
    - `source venv/bin/activate`
 4. Navigate to the projects directory `cd ./End-to-End-Gen-AI-Powered-App` using your terminal
 5. Install Libraries: `pip install -r requirements.txt`
 6. Navigate to the app directory `cd ./app` using your terminal 
 7. run `streamlit run app.py`
 8. open the link displayed in the terminal on your preferred browser

## Collaboration
- Collaborations are welcomed ❤️

## Acknowledgments
 - I would like to thank [Krish Naik](https://www.youtube.com/@krishnaik06)
   
## Contact
 - LinkedIn: [Nebeyou Musie](https://www.linkedin.com/in/nebeyou-musie)
 - Gmail: nebeyoumusie@gmail.com
 - Telegram: [Nebeyou Musie](https://t.me/NebeyouMusie)


