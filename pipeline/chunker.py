from schemas.problem import Problem



def build_problem_chunk(
        ticket,
        problem
):


    text=f"""

问题:

{problem.problem_type}


分类:

{" > ".join(problem.category)}


典型表现:

{"；".join(problem.symptom)}


影响:

{"；".join(problem.impact)}


地点类型:

{problem.location_type}


关键词:

{"；".join(problem.keywords)}

"""


    return {


        "id":
        ticket.ticket_id,


        "type":
        "problem",


        "text":
        text,


        "metadata":{


            "city":
            ticket.city,


            "district":
            ticket.district,


            "category3":
            ticket.category3

        }

    }