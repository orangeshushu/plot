import requests
import csv

# 定义一个函数用于调用API
def generate_tweets(prompt, num_tweets):
    url = "https://api.openai.com/v1/engines/text-davinci-002/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    tweets = []
    for _ in range(num_tweets):
        data = {
            "prompt": prompt,
            "max_tokens": 1024,
            "n": 1,
            "stop": None,
            "temperature": 0.8,
        }

        response = requests.post(url, headers=headers, json=data)
        if "choices" in response.json():
            tweet = response.json()["choices"][0]["text"].strip()
            print(tweet)
            tweets.append(tweet)
        else:
            print(f"Error: {response.text}")
    return tweets

# 获取API_KEY
API_KEY = "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"

# 生成正样本和负样本
positive_prompt = "Generate tweet data with the requirement that the content must describe self-recovery from COVID-19, using first-person perspective. Please use multiple ways to express it and avoid similarity between the contents."
negative_prompt = "Generate some random tweet data, but the content cannot be about self-recovery from COVID-19."
num_samples = 100

positive_tweets = generate_tweets(positive_prompt, num_samples)
negative_tweets = generate_tweets(negative_prompt, num_samples)

# 将正样本保存到recover.csv
with open("recover.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["tweet"])
    for tweet in positive_tweets:
        writer.writerow([tweet])

# 将负样本保存到other.csv
with open("other.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["tweet"])
    for tweet in negative_tweets:
        writer.writerow([tweet])
