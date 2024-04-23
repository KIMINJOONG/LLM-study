# -*- coding: utf-8 -*-
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    load_dotenv()
    print("Hello langchain")
    information = """
    남아프리카 공화국, 캐나다, 미국 삼중국적이며, 테슬라, 스페이스X, X(구 트위터)를 필두로 다수의 기술 기업을 경영하고 있다. 2024년 기준 전 세계에서 가장 부유한 인물이기도 하다. 혁신의 대명사로 호평받는 동시에 여러 기행으로 구설수에 오르고 비판받으면서 매순간의 언행과 결단에 대해 세계의 주목을 받고 있다.
    2018년 9월 7일 코미디언 조 로건이 진행하는 생방송 팟캐스트인 '조 로건 익스피리언스'에 출연해서 마리화나가 섞인 담배를 피우고 위스키를 마셔서 논란이 됐다. 단, 직접 불을 붙여서 태운 것은 아니며 진행자가 피우고 있던 마리화나를 권유에 의해서 한 모금 흡입한 게 전부였다. 또한 머스크는 진행자에게 합법적인 것이냐고 물어보고 흡입했다. 해당 장소가 마리화나가 합법화된 주이며 미 서부에서 마리화나의 위치는 담배와 술이랑 크게 차이가 안 나는 수준에 속하긴 해도, 저명한 재계 인사가 공개적으로 마리화나를 피웠다는 사실만으로도 논란을 일으키기에 충분했다. 머스크가 일반적인 기업 경영자였다면 가십거리 정도로 끝났을 일이지만, 문제는 그가 경영하는 스페이스X가 미국 공군과 사업 계약을 맺은 상태라는 점이다. 미군 규정에 따라 미군과 계약한 회사 직원은 마리화나 흡연이 금지되어 있기 때문에 이 문제로 또다시 조사를 받아야 할 처지가 됐다. 그리고 결국 미 공군이 2021년부터 사용할 발전형 소모성 발사체(Evolved Expendable Launch Vehicle) 사업 계약은 스페이스X 대신 노스롭 그루먼의 오메가 로켓이 선정되었다. 블룸버그는 국방부가 스페이스X의 비밀 취급 인가를 재검토하고 있다고 보도했다.

    2019년 2월에는 릭 앤 모티의 저스틴 로일랜드와 함께 PewDiePie의 'Meme Review' 영상에 출연하면서 인터넷을 한바탕 뒤집어 놨다. 이미 1월달에 이를 암시하는 트윗을 올리긴 했지만 당시는 다들 그냥 드립으로 넘어갔던 터라 다시금 머스크의 어디로 튈지 모르는 행보에 혀를 내두르게 만들었다.

    2019년 3월 31일[40]에는 사운드클라우드에 고릴라 하람베에 대한 랩을 올리며[41] 사운드클라우드 래퍼에 등극했다. 다른 사람이었다면 때이른 만우절 장난이라고 여기겠지만 워낙 쓸데없이 고퀄리티인데다[42] 사람이 사람이다 보니 다들 '역시 머스크'라는 반응이다. 그 이후로도 Don't Doubt ur Viv    e라는 곡을 올렸다.

    2019년 4월에는 트위터 프사를 'Absolute Unit' 밈으로 유명한 양 사진으로 바꿨다가 또다시 뜬금없이 에드워드 엘릭 스크린샷으로 바꿨다.

    2019년 6월에는 프로필 사진을 검은색으로 바꾸고 '트위터 계정을 삭제했다'는 트윗을 올리더니 닉네임을 'Daddy DotCom'으로 바꿨다. 이때 트위터 계정 삭제 소식에 테슬라 주가가 일시적으로 급등하기도 했다. 물론 며칠 뒤에는 언제 그랬냐는 듯이 화성에서 로드스터 보닛에 앉아 커피를 즐기는 스타맨#으로 프로필 사진을 바꾸고 다시 뻘글을 써대기 시작했다.
    """
    summary_template ="""
    이 사람에 대한 정보{information}를 가지고 두가지를 만들어줘
    1. 간단한 요약
    2. 그 사람에 대한 두가지 흥미로운 사실
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)


    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    res = chain.invoke(input={"information": information})

    print(res)
