from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_python_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.types import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool

load_dotenv()


def main():
    print("Start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    python_agent_executor.run("""generate and save in current working directory 15 QRcodes
                                 that point to www.udemy.com/course/langchain, you have qrcode package installed already""")


if __name__ == "__main__":
    main()
