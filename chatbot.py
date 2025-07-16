"""
Chat with Pre-processed PDF
"""

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import torch

from config import *


class FastMathChatbot:
    def __init__(self, model_name=DEFAULT_MODEL, vectorstore_path=DEFAULT_VECTORS):
        """Initialize chatbot with saved vector store"""
        print(f"Starting chatbot with model: {model_name}")
        
        # Check if vector store exists
        if not os.path.exists(vectorstore_path):
            raise FileNotFoundError(f"Vector store not found at: {vectorstore_path}")
        
        # Setup memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Load embeddings (same as used for processing)
        print("Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Load vector store
        print("Loading vector store...")
        self.vectorstore = FAISS.load_local(
            vectorstore_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Setup LLM
        print("Connecting to model...")
        self.llm = Ollama(
            model=model_name,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS
        )
        
        # Create chat chain
        print("Setting up chat chain...")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True
        )
        
        print("Chatbot ready!")
    
    def ask(self, question):
        """Ask a question"""
        # Get response
        response = self.chat_chain.invoke({"question": question})
        
        # Format answer
        answer = response["answer"]
        sources = response.get("source_documents", [])
        
        print(f"\nQuestion: {question}")
        print(f"\nAnswer: {answer}")
        
        # Show sources
        if sources:
            print(f"\nSources ({len(sources)} sections):")
            for i, doc in enumerate(sources[:], 1):
                page = doc.metadata.get("page", "?")
                content = doc.page_content[:]
                print(f"  {i}. Page {page}: {content}")
        
        return answer
    
    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()
        print("Memory cleared!")
    
    def test_connection(self):
        """Test if model is working"""
        try:
            response = self.llm.invoke("What is 2+2?")
            print(f"Model test: {response}")
            return True
        except Exception as e:
            print(f"Model test failed: {e}")
            return False