# TODO Add some test
from transformers import BertTokenizer, AlbertForSequenceClassification

model_path = "clhuang/albert-news-classification"
model = AlbertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# Category index
news_categories = ["政治", "科技", "運動", "證卷", "產經", "娛樂", "生活", "國際", "社會", "文化", "兩岸"]
idx2cate = {i: item for i, item in enumerate(news_categories)}


# get category probability
def get_category_proba(text):
    max_length = 250
    # prepare token sequence
    inputs = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # perform inference
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)

    # executing argmax function to get the candidate label index
    label_index = probs.argmax(dim=1)[0].tolist()  # convert tensor to int
    # get the label name
    label = idx2cate[label_index]

    # get the label probability
    proba = round(float(probs.tolist()[0][label_index]), 2)

    response = {"label": label, "proba": proba}

    return response


get_category_proba("俄羅斯2月24日入侵烏克蘭至今不到3個月，芬蘭已準備好扭轉奉行了75年的軍事不結盟政策，申請加入北約。芬蘭總理馬林昨天表示，「希望我們下星期能與瑞典一起提出申請」。")
