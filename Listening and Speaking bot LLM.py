import os;

os.environ['OPENAI_API_KEY']='Add your openAI API key'
os.environ['SERPAPI_API_KEY']='Add your SerpAPI Key '

import pyttsx3
import speech_recognition as sr


from langchain.llms import OpenAI



def speak(text): 
    engine = pyttsx3.init()
    id='HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0'
    engine.setProperty("voices",id)
    engine.say(text=text)
    engine.runAndWait()

speak("Hello sir")




def speechrecognition():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening")
        r.pause_threshold=1
        audio=r.listen(source,0,8)

    try:
        print("Recognizing")
        query=r.recognize_google(audio,language="en")
        return query.lower()

    except:
        return ""

speech=speechrecognition()
print(speech)



from langchain.prompts import PromptTemplate



llm=OpenAI(temperature=0.6)
from langchain.chains import LLMChain
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food,suggest me a name for the restaurant"
)


name_chain=LLMChain(llm=llm,prompt=prompt_template_name,output_key="restaurant_name")

prompt_template_item=PromptTemplate(
    input_variables=['restaurant_name'],
    template="Suggest me some menu for{restaurant_name},separate with commas"
)
food_items_chain=LLMChain(llm=llm,prompt=prompt_template_item,output_key="menu_items")



from langchain.chains import SequentialChain

chain=SequentialChain(
    chains=[name_chain,food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name','menu_items']
)
chain({'cuisine':speech})


from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.llms import OpenAI

speech=speechrecognition()
print(speech)
llm=OpenAI(temperature=0.1)
tools=load_tools(["wikipedia","serpapi","llm-math"],llm=llm)
agent=initialize_agent(
     tools,
     llm,
     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
     )

sp=agent.run(speech)
print(sp)
speak(sp)


