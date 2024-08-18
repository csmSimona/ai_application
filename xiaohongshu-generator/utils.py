from prompt_template import system_template_text, user_template_text
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser # 用于指挥AI输出符合格式要求的JSON，并且帮我们解析
from xiaohongshu_model import Xiaohongshu

def generate_xiaohongshu(theme, api_key, base_url):
    # 1、创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template_text),
        ("user", user_template_text)
    ])

    # 2、创建模型
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, base_url=base_url)

    # 3、创建输出解析器（入参为自定义的数据模型）
    output_parser = PydanticOutputParser(pydantic_object=Xiaohongshu)

    # 4、创建链
    chain = prompt | model | output_parser

    # 5、调用链
    result = chain.invoke({
        "parser_instructions": output_parser.get_format_instructions(), # 给AI的输出格式指令
        "theme": theme
    })

    return result

# print(generate_xiaohongshu("奥运会", "sk-15n9nCbS08o5OajA5RPMxegzbd7UPFmWxciSQuaksrx2OFos", "https://api.chatanywhere.tech/v1"))
