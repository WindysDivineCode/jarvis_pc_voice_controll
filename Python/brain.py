from ollama import chat

response = chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': 'i will name you jarvis for a project '
        'i want to make so from now on answer to jarvies'}
    ]
)

print(response['message']['content'])