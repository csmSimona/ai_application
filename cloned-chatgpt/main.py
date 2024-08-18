
import streamlit as st
from utils import get_chat_response
from langchain.memory import ConversationBufferMemory


st.title("💬 克隆ChatGPT")

# 创建侧边栏
with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    base_url = st.text_input("请输入OpenAI baseurl：")
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-keys)")

# 创建会话状态
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
    st.session_state["messages"] = [{"role": "ai",
                                     "content": "你好，我是你的AI助手，有什么可以帮你的吗？"}]
    
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])


prompt = st.chat_input("请输入你的问题")

submit = st.button("清空对话")

if prompt:
    if not openai_api_key:
        st.info("请输入你的OpenAI API密钥")
        st.stop()
    st.session_state["messages"].append({"role": "human", "content": prompt})
    st.chat_message("human").write(prompt)

    with st.spinner("正在思考..."):
        response = get_chat_response(prompt, st.session_state["memory"], openai_api_key, base_url)
        st.session_state["messages"].append({"role": "ai", "content": response})
        st.chat_message("ai").write(response)

if submit:
    st.session_state["memory"].clear()
    st.session_state["messages"] = [{"role": "ai",
                                     "content": "你好，我是你的AI助手，有什么可以帮你的吗？"}]
    st.rerun()

