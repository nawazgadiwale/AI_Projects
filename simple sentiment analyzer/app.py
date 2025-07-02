import gradio as gr 
from transformers import pipeline

# loading the pretrained model from transformers 
sentiment_analyzer = pipeline('sentiment-analysis')

# funtion to anlayze the text 
def predict_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = round(result['score'],3)
    return f"sentiment : {label} ({score})"

#testing from text file 
def test_from_file(filename = "test_text.txt"):
    try:
        with open(filename,"r",encoding="utf-8") as f:
            print("\n Runnnig the test selently : ")
            for line in f:
                text = line.strip()
                if text:
                    result = predict_sentiment(text)
                    print(f"input : {text}\n {result}\n")
    except FileNotFoundError:
        print(f"file '{filename}' not found")
                    

if __name__ == '__main__':
    test_from_file()

    # gradio interface 
    gr.Interface(
    fn=sentiment_analyzer,
    inputs=gr.Textbox(lines=4, label="Enter your sentence"),
    outputs="text",
    title="🧠 Mini BERT Sentiment Analyzer",
    description="Enter a sentence to check its sentiment using BERT",
    examples=[
        ["I love this movie!"],
        ["This app is terrible."],
        ["The experience was average, not great."],
    ]
).launch()
