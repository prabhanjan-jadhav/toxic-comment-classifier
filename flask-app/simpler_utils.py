# Importing necessary libraries
import numpy as np
import pandas as pd
import re
# Model Building
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset

model = torch.load('dsbert_toxic.pt', map_location=torch.device('cpu'))

# get predictions
def get_predictions(comment):

    comment_input = []
    comment_input.append(comment)
    test_df = pd.DataFrame()
    test_df['comment_text'] = comment_input
    cols = {'toxic':[0], 'severe_toxic':[0], 'obscene':[0], 'threat':[0], 'insult':[0], 'identity_hate':[0]}
    for key in cols.keys():
        test_df[key] = cols[key]
    
    # Data Cleaning and Preprocessing
    cleaned_data = test_df.copy()
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"https?://\S+|www\.\S+","",x) )
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub("["
                                                                                       u"\U0001F600-\U0001F64F"
                                                                                       u"\U0001F300-\U0001F5FF"  
                                                                                       u"\U0001F680-\U0001F6FF" 
                                                                                       u"\U0001F1E0-\U0001F1FF"  
                                                                                       u"\U00002702-\U000027B0"
                                                                                       u"\U000024C2-\U0001F251"
                                                                                       "]+","", x, flags=re.UNICODE))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",x))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"<.*?>","",x))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\"\"", "\"",x))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"^\"", "",x))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\"$", "",x))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"[^a-zA-Z0-9\s][^a-zA-Z0-9\s]+", " ",x))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"[^a-zA-Z0-9\s\"\',:;?!.()]", " ",x))
    cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\s\s+", " ",x))
    Final_data = cleaned_data.copy()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    class Toxic_Dataset(Dataset):
        def __init__(self, Comments_, Labels_):
            self.comments = Comments_.copy()
            self.labels = Labels_.copy()

            self.comments["comment_text"] = self.comments["comment_text"].map(lambda x: tokenizer(x, padding="max_length", truncation=True, return_tensors="pt"))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            comment = self.comments.loc[idx,"comment_text"]
            label = np.array(self.labels.loc[idx,:])

            return comment, label

    X_test = pd.DataFrame(test_df.iloc[:, 0])
    Y_test = test_df.iloc[:, 1:]
    Test_data = Toxic_Dataset(X_test, Y_test)
    Test_Loader = DataLoader(Test_data, shuffle=False)

    for comments, labels in Test_Loader:
        labels = labels.to('cpu')
        labels = labels.float()
        masks = comments['attention_mask'].squeeze(1).to('cpu')
        input_ids = comments['input_ids'].squeeze(1).to('cpu')

        output = model(input_ids, masks)
        op = output.logits

        preds = []
        for i in range(6):
            preds.append(round(op[0, i].tolist(), 2))

    return preds