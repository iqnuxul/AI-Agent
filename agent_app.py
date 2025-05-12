import os
import json
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from model_tool import predict_sales

os.environ["OPENAI_API_KEY"] = "I don't have OpenAI API Key yet"

# ========== LangChain Tool ==========
predict_sales_tool = Tool(
    name="Sales Predictor",
    func=lambda x: str(predict_sales(eval(x))),
    description=(
        "Predict daily sales based on various store features."
        "The input must be a Python dictionary containing the following fields:"
        "'Store', 'Day', 'Month', 'Year', 'WeekOfYear', 'StoreType', 'Assortment', 'Promo', 'Promo2', "
        "'Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval', 'SchoolHoliday', "
        "'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', "
        "'StateHoliday', 'Open'.\n"
        "For example: {'Store': 300, 'Day': 12, 'Month': 5, 'Year': 2015, 'WeekOfYear': 19, "
        "'StoreType': 'a', 'Assortment': 'a', 'Promo': 1, 'Promo2': 1, 'Promo2SinceYear': 2011, "
        "'Promo2SinceWeek': 13, 'PromoInterval': 'Feb,May,Aug,Nov', 'SchoolHoliday': 0, "
        "'CompetitionDistance': 150.0, 'CompetitionOpenSinceMonth': 6, 'CompetitionOpenSinceYear': 2012, "
        "'StateHoliday': '0', 'Open': 1}"
    )
)

# ========== Initialize Language Model ==========
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4"  # You can also use "gpt-3.5-turbo"
)

# ========== Initialize Agent ==========
agent = initialize_agent(
    tools=[predict_sales_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ========== Conversation with the User ==========
if __name__ == "__main__":
    print("Welcome to the Rossmann Store Sales Prediction Agent! Enter your question, or type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            response = agent.run(user_input)
            print("\nAI Agent: ", response)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
