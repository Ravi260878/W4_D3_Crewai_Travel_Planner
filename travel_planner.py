import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# 1. Explicitly load the .env in the same folder as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# 2. Ask user for a city
city = input("Which city? ")

# 3. Configure the LLM with gpt-4o-mini
llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# 4. Create the Agent
planner = Agent(
    role="Expert Local Travel Planner",
    goal="Suggest exactly 3 top places to visit in {city} with practical, actionable tips.",
    backstory=(
        "You are an experienced local travel curator who knows the essence of every city. "
        "You provide concise, high-value recommendations so visitors can explore the best "
        "highlights without getting overwhelmed."
    ),
    llm=llm,
    verbose=True
)

# 5. Create the Task
task = Task(
    description=(
        "Suggest exactly 3 places to visit in {city}. "
        "For each place, provide its name and a concise one-line insider tip or key feature."
    ),
    expected_output=(
        "A clean numbered list of exactly 3 places in this format:\n"
        "1. [Place Name]\n"
        "[Brief description / tip]\n"
        "2. [Place Name]\n"
        "[Brief description / tip]\n"
        "3. [Place Name]\n"
        "[Brief description / tip]"
    ),
    agent=planner
)

# 6. Assemble the Crew
crew = Crew(
    agents=[planner],
    tasks=[task],
    verbose=True
)

# 7. Kick off the execution and print the final result
result = crew.kickoff(inputs={"city": city})
print("\n" + "=" * 40 + "\n")
print(result)