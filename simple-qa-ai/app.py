from transformers import pipeline

qa_pipeline = pipeline("question-answering")

def main():
    print(" Terminal-based BERT QA System\n")
    
    context = input("Enter context paragraph:\n")
    question = input("\n Enter your question:\n")
    
    result = qa_pipeline({
        'context': context,
        'question': question
    })
    
    print(f"\n Answer: {result['answer']}")
    print(f" Confidence: {round(result['score'], 3)}")

if __name__ == "__main__":
    main()
