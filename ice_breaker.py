from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_profile_url
from third_parties.linkedin import scrape_linkedin_profile


if __name__ == "__main__":
    print("Hello from LangChain")
    linkedin_profile_url = linkedin_profile_url(name="Nemsiss Shahbazian")
    summary_template = """
        given the information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about them
     """

    # we specify that the input variable is called information
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # the temperature determines how create the model will be, 0 means it wont be creative
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # what runs the llm is the chain
    # the first arguement (llm) is actually the chatmodel which has the llm inside it
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))
