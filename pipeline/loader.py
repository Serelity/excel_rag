import pandas as pd

from schemas.ticket import Ticket



def safe_str(value):

    if pd.isna(value):

        return ""

    return str(value).strip()



def load_tickets(path):


    df=pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )


    for _,row in df.iterrows():


        yield Ticket(

            ticket_id=safe_str(
                row.get("case_id")
            ),


            content=safe_str(
                row.get("case_content")
            ),


            goal=safe_str(
                row.get("case_goal")
            ),


            category1=safe_str(
                row.get(
                    "case_accord_type_one_name"
                )
            ),


            category2=safe_str(
                row.get(
                    "case_accord_type_two_name"
                )
            ),


            category3=safe_str(
                row.get(
                    "case_accord_type_three_name"
                )
            ),


            city=safe_str(
                row.get(
                    "area_code_city"
                )
            ),


            district=safe_str(
                row.get(
                    "area_code_area"
                )
            ),


            create_time=safe_str(
                row.get(
                    "call_time"
                )
            )

        )