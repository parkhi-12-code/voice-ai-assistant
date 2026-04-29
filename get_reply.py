from groq import Groq

def get_reply(user_text):
    print("🤔 Thinking of a reply...")
    
    client = Groq(api_key="YOUR_GROQ_KEY_HERE")
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": user_text}
        ], 
        model="llama-3.3-70b-versatile",
    )
    
    reply = chat_completion.choices[0].message.content
    print(" AI says:", reply)
    return reply

# Test it
get_reply("What is machine learning in 2 sentences?")