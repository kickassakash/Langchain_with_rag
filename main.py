from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="gemma3:4b-it-qat")

template = '''
you are an expert in answering questions about pizza restraunts:

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n")
    question = input("enter your query(type quit to exit): ")
    if question.lower() == "quit":
        break
    print("\n\n")
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews":reviews,"question":question})
    print(result)