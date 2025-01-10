from models import load_model
from vector_store import load_vector_store
from langchain.chains import RetrievalQA

def main():
    print("Загружаем базу знаний...")
    retriever = load_vector_store("knowledge_base.txt")

    print("Загружаем модель...")
    llm = load_model()

    print("Создаем RAG pipeline...")
    qa = RetrievalQA(llm=llm, retriever=retriever)

    print("\nСистема готова! Задавайте вопросы.")
    while True:
        query = input("\nВведите ваш вопрос (или 'exit' для выхода): ")
        if query.lower() == 'exit':
            print("Завершаем работу.")
            break
        answer = qa.run(query)
        print(f"Ответ: {answer}")

if __name__ == "__main__":
    main()
