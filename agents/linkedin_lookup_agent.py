from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
    Answer with actual url. No preface."""
    # the name of the tools need to be unique, the description for each tool has to be descriptive and accurate since the agent will use that to decide whether to use the tool or not
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Linkedin profile page",
            func=get_profile_url,
            description="useful for when you need to get the Linkedin profile URL",
        )
    ]
    # if agent type is not passed, it will by default be the zero-shot react
    # verbose=True makes the agent tell us what it does at each step that it takes, this way, we will see the reasoning process of that agent
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
