import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent


st.title("📑 AI智能PDF问答工具")

# 创建侧边栏
with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    base_url = st.text_input("请输入OpenAI baseurl：")
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-keys)")

# 把memory存储到会话状态
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

uploaded_file = st.file_uploader("上传PDF文件：", type="pdf")
question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("请输入OpenAI API密钥")

if uploaded_file and question and openai_api_key:
    with st.spinner("正在处理中，请稍等..."):
        response = qa_agent(openai_api_key, base_url, st.session_state["memory"], uploaded_file, question)
        print(response)
    
    st.write("### 答案")
    st.write(response["answer"])
    # 存储历史消息
    st.session_state["chat_history"] = response["chat_history"]

if st.session_state.get("chat_history"):
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(f"人类: {human_message.content}")
            st.write(f"AI: {ai_message.content}")
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
