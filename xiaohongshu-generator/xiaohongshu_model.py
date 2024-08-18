# BaseModel：创建数据模式；Field：为BaseModel里的数据提供额外信息和验证条件
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List # 从类型库引入类型名List

# 定义数据模型
# 继承BaseModel的类
class Xiaohongshu(BaseModel):
    # 定义字段，补充描述信息
    titles: List[str] = Field(description="小红书的5个标题", min_items=5, max_items=5)
    content: str = Field(description="小红书的正文内容")