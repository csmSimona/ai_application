from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def qa_agent(openai_api_key, base_url, memory, uploaded_file, question):
    # 创建模型
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, base_url=base_url)

    # 读取上传的文件文档内容存储到本地，再使用PyPDFLoader加载
    file_content = uploaded_file.read() # 返回二进制内容
    temp_file_path = "temp.pdf"
    # 写入本地文件中
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    # 创建pdf文档加载器
    loader = PyPDFLoader(temp_file_path)
    # 加载文档内容
    docs = loader.load()

    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 每块文本的最大长度
        chunk_overlap=40, # 分割片段之间重叠的长度
        separators=["\n", "。", "！", "？", "，", "、", ""] # 用于分割的字符
    )
    # 分割文本
    texts = text_splitter.split_documents(docs)

    # 创建嵌入模型
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key, base_url=base_url)
    # 创建向量数据库
    db = FAISS.from_documents(texts, embeddings_model)
    # 创建检索器
    retriever = db.as_retriever()

    # 创建带记忆的检索增强生成对话链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    # 调用得到回答
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
