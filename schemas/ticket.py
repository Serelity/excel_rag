from pydantic import BaseModel


class Ticket(BaseModel):
    """
    一个ticket包含的内容，将每一行数据转化为一个ticket对象
    """
    ticket_id:str

    content:str

    goal:str

    category1:str

    category2:str

    category3:str

    city:str

    district:str

    create_time: str = ""
