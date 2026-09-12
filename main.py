import yaml
import json


from pipeline.loader import load_tickets
from pipeline.cleaner import clean_ticket
from pipeline.extractor import ProblemExtractor
from pipeline.chunker import build_problem_chunk



with open(
    "configs/config.yaml",
    "r",
    encoding="utf8"
) as f:

    config=yaml.safe_load(f)



extractor=ProblemExtractor(

    config["llm"]

)



output=config["data"]["output"]



with open(
    output,
    "w",
    encoding="utf8"
) as f:


    for ticket in load_tickets(
        config["data"]["input"]
    ):


        ticket=clean_ticket(
            ticket
        )


        problem=extractor.extract(
            ticket
        )


        chunk=build_problem_chunk(

            ticket,

            problem

        )


        f.write(

            json.dumps(
                chunk,
                ensure_ascii=False
            )
            +
            "\n"

        )


print(
    "Problem Chunk生成完成"
)