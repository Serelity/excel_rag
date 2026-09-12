import json

from openai import OpenAI


from schemas.problem import Problem



SYSTEM_PROMPT="""

你是城市治理工单分析专家。


任务：

从群众投诉中抽取标准化治理问题。


要求：

problem_type:
输出问题名称，例如：
占道经营
噪音扰民
垃圾处理


symptom:
群众描述的现象


impact:
造成影响


location_type:
地点类型


keywords:
关键词


只输出JSON。

"""



class ProblemExtractor:


    def __init__(
        self,
        config
    ):


        self.client=OpenAI(

            base_url=
            config["base_url"],

            api_key="EMPTY"

        )


        self.model=config["model"]



    def extract(
        self,
        ticket
    ):


        prompt=f"""

{SYSTEM_PROMPT}


分类:

{ticket.category1}

{ticket.category2}

{ticket.category3}


描述:

{ticket.content}


诉求:

{ticket.goal}

"""


        response=self.client.chat.completions.create(

            model=self.model,


            temperature=0,


            messages=[

                {
                    "role":"system",
                    "content":
                    "你是专业数据分析模型"
                },


                {
                    "role":"user",
                    "content":prompt
                }

            ]

        )


        text=response.choices[0].message.content


        data=json.loads(text)


        return Problem(
            **data
        )