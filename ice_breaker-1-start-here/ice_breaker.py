# -*- coding: utf-8 -*-
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("Hello Langchain!")
    summary_template ="""
    이 링크드인에 대한 정보{information}를 가지고 두가지를 만들어줘
    1. 간단한 요약
    2. 그것에 대한 두가지 흥미로운 사실
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)


    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/harrison-chase-961287118/"
    )

    print(chain.invoke(input={"information": linkedin_data}))
