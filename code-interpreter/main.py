from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents.types import AgentType
from langchain_experimental.agents.agent_toolkits import (
    create_python_agent,
    create_csv_agent,
)
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.initialize import initialize_agent
from langchain.agents import Tool


load_dotenv()


def main():
    print("Start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # python_agent_executor.run(
    #     """generate and save in current working directory 15 QRcodes
    #                              that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    # )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="/Users/kiminjoong/LLM-study/code-interpreter/episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent.run("how many columns are there in file episode_info.csv")
    # csv_agent.run("in file episode_info.csv, which writer wrote the least episodes? how many episodes did he write?")
    # csv_agent.run("print seasons ascending order of the number of episodes they have")

    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""useful when you need to transform natural language and write from it python and execute the python code returning the results of the code execution, DO NOT SEND PYTHON CODE TO THIS TOOL""",
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent.run,
                description="""useful when you need to answer question over episode_info.csv file, takes an input the entire question and returns the answer after running pandas calculations""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    # grand_agent.run(
    #     """generate and save in current working directory 15 QRcodes
    #                              that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    # )

    grand_agent.run("print seasons ascending order of the number of episodes they have")


if __name__ == "__main__":
    main()
