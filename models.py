from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

def load_model():
    # Необходимо предварительно залогиниться в терминале на huggingface-cli и ввести свой токен
    model_name = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    return llm
