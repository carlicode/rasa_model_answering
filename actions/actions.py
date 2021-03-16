import datetime as dt 
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from transformers import pipeline

class ActionHelloWorld(Action):

    nlp = pipeline(
    'question-answering', 
    model='mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',
    tokenizer=(
        'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',  
        {"use_fast": False}
    ))
    context =  'Frédéric François Chopin[nota 1]​ (en polaco Fryderyk Franciszek Chopin;[nota 2]​ Żelazowa Wola, Gran Ducado de Varsovia, 1 de marzo[1]​[2]​[nota 3]​ de 1810-París, 17 de octubre de 1849) fue un profesor, compositor y virtuoso pianista polaco, considerado uno de los más importantes de la historia y uno de los mayores representantes del Romanticismo musical.[3]​[4]​[5]​[6]​ Su maravillosa técnica, su refinamiento estilístico y su elaboración armónica se han comparado históricamente, por su influencia en la música posterior, con las de Wolfgang Amadeus Mozart, Ludwig van Beethoven, Johannes Brahms, Franz Liszt o Serguéi Rajmáninov.'
    question = '¿Quién fue Frédéric François Chopin? '
    model = nlp({
        'question': question,
        'context': context})
    answer = model['answer']

    def name(self) -> Text:
        return "answer_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="answer")

        return []
