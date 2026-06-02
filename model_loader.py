from transformers import pipeline

def load_model():
    print("Loading model... ⏳")
    llm = pipeline(
        "text-generation",
        model="openai-community/gpt2",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )
    print("Model ready! ✅")
    return llm