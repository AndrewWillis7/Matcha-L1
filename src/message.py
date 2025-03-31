import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Message:
    def __init__(self, text):
        self.text = text
        self.word_count = self._compute_word_count()
        self.sentiment = self._compute_sentiment()

    def _compute_word_count(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokens = tokenizer.tokenize(self.text)
        return len(tokens)

    def _compute_sentiment(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        inputs = tokenizer(self.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        sentiment_label = "positive" if predicted_class == 1 else "negative"
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        sentiment_score = probabilities[predicted_class].item()

        return {
            "label": sentiment_label,
            "score": sentiment_score
        }

    def save_to_json(self, filename="messages.json"):
        message_data = {
            "text": self.text,
            "word_count": self.word_count,
            "sentiment": self.sentiment
        }

        # Check if file exists and load existing data
        if os.path.exists(filename):
            with open(filename, "r") as file:
                data = json.load(file)
        else:
            data = []

        # Check for duplicates before appending
        if not any(msg["text"] == self.text for msg in data):
            data.append(message_data)
            with open(filename, "w") as file:
                json.dump(data, file, indent=4)
            print("Message saved.")
        else:
            print("Duplicate message detected. Skipping save.")

    def __str__(self):
        return (
            f"Message: {self.text}\n"
            f"Word Count: {self.word_count}\n"
            f"Sentiment: {self.sentiment['label']} (Score: {self.sentiment['score']:.2f})"
        )

# Example Usage
#if __name__ == "__main__":
    #msg1 = Message("I love Python and PyTorch!")
    #msg2 = Message("This is not very good.")
    #msg3 = Message("I love Python and PyTorch!")  # Duplicate message

    #msg1.save_to_json()  # Saves to messages.json
    #msg2.save_to_json()  # Appends to the same file
    #msg3.save_to_json()  # Duplicate, will not be saved

    #print(msg1)
    #print("\nData saved to 'messages.json'")
