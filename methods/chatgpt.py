import openai

# Replace with your API key
api_key = "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"

openai.api_key = api_key

# Define the prompt for ChatGPT
prompt = "请生成50条描述自我从新冠中康复的英文推特推文"

# Define the model you want to use, e.g., "text-davinci-002"
model_name = "text-davinci-002"

# Make the API call to ChatGPT
response = openai.Completion.create(
    engine=model_name,
    prompt=prompt,
    max_tokens=500,
    n=1,
    stop=None,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Extract the generated response from the API call
generated_response = response.choices[0].text.strip()

# Print the generated response
print("ChatGPT Response:\n", generated_response)