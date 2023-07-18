from typing import Tuple

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from agents.linkedin_lookup_agent import lookup
from output_parsers import person_intel_parser, PersonIntel

from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from agents.twitter_lookup_agent import lookup as twitter_lookup

name = "Eden Marco"


def ice_break(name: str) -> Tuple[PersonIntel, str]:
    linkedin_profile_url = lookup(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    twitter_username = twitter_lookup(name=name)
    tweets = scrape_user_tweets(username=twitter_username)
    summary_template = """
            given the Linkedin information {linkedin_information} and twitter {twitter_information} about a person I want you to create:
            1. a short summary
            2. two interesting facts about them
            3. a topic that may interest them
            4. 2 creative ice breakers to start a conversation with them
            \n{format_instructions} 
         """

    # we specify that the input variable is called information
    # the "information" key corresponds to the {information} in summary template
    # the partial variables is for formatting, so that we can format the LLM's response in either JSON or any other format and prepare it to send it back as a response to a frontend app
    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    # the temperature determines how create the model will be, 0 means it won't be creative
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # what runs the llm is the chain
    # the first arguement (llm) is actually the chatmodel which has the llm inside it (since chatmodels are wrappers around LLMs)
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    result = chain.run(linkedin_information=linkedin_data, twitter_information=tweets)
    # print(result)

    # return the parsed object
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    print("Hello from langchain!")
    ice_break(name="Eden Marco")
