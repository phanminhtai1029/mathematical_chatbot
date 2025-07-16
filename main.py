from chatbot import FastMathChatbot
from config import *

def main():
    """Main function"""
    print("=" * 50)
    print("FAST MATH CHATBOT")
    print("=" * 50)
    
    # Set paths
    vectorstore_path = DEFAULT_VECTORS  # Where you saved the vector store
    model_name = DEFAULT_MODEL  # Change if you want different model
    
    # Create chatbot
    try:
        chatbot = FastMathChatbot(model_name, vectorstore_path)
    except Exception as e:
        print(f"Error creating chatbot: {e}")
        return
    
    # Test connection
    if not chatbot.test_connection():
        print("Model connection failed!")
        return
    
    # Chat loop
    print("\nReady! Ask math questions or type 'quit' to exit")
    print("Commands: 'quit', 'clear' (clear memory)")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'quit':
                print("Goodbye!")
                break
            
            if question.lower() == 'clear':
                chatbot.clear_memory()
                continue
            
            # Ask question
            chatbot.ask(question)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()