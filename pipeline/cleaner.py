import re



def clean_text(text):


    if not text:
        return ""


    text=text.replace(
        "\n",
        " "
    )


    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()



def clean_ticket(ticket):


    ticket.content = clean_text(
        ticket.content
    )


    ticket.goal = clean_text(
        ticket.goal
    )


    return ticket