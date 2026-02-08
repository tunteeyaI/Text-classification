import requests
from config import APIKEY
#set up the models
APIURL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
HEADERS={"Authorization":f" Bearer {APIKEY}"}
TOPICS={"Math","Sports","News","Science"}
def classifytext(text):
    #prepare the payloaf for the API request
    payload={
        "inputs": {
            "text": text,
            "labels": TOPICS
        }
    }
    #make the API request
    response = requests.post(APIURL, headers=HEADERS,json=payload)
    #check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        #get the predicted label with the highest score
        predicted_label = response_data[0]['label']
        return predicted_label
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
#Ask the user for input text
user_input = input("Enter a text to classify:")
#Classify the text
predicted_topic = classifytext(user_input)
if predicted_topic:
    print(f"The predicted topic is: {predicted_topic}")
else:
    print("Could not classify the text.")