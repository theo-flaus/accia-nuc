from transformers import pipeline

class ExtractiveModel:

    def __init__(self, model : str, tokenizer : str) -> None:
        """Instancy object of class Model, given a model and a tokenizer from hugging face
            (This has to be an extractive QnA model)

        Args:
            model (str): model name in hugging face
            tokenizer (str): tokenizer name in hugging face
        """
        self.nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    def get_answer(self, question : str, context: str) -> dict:
        """Performs extractive QnA by providing an answer to the question asked, given a context.

        Args:
            question (str): the question asked
            context (str): a small portion of text thaht contains the anwser to the question

        Returns:
            dict: the answer with its probability score and its start and end position in the given context
        """
        return self.nlp({'question': question, 'context': context})



class GenerativeModel:

    def __init__(self) -> None:
        self.text2text_generator = pipeline("text2text-generation")

    def get_answer(self, question : str, context : str) -> dict:
        """_summary_

        Args:
            question (str): _description_
            context (str): _description_

        Returns:
            _type_: _description_
        """
        return self.text2text_generator(f'question : {question} context : {context}')[0]

    def generate_answer(self, question : str, context: str) -> dict:
        """_summary_

        Args:
            question (str): _description_
            context (str): _description_

        Returns:
            dict: _description_
        """
        prompt = f'Trouve la réponse à cette question : {question}\nDans le contexte suivant : {context}'
        return self.text2text_generator(prompt)[0]

    def reformulate_question(self, question : str) -> dict:
        prompt = f'Reformule la question : {question}'
        return self.text2text_generator(prompt)[0]

    