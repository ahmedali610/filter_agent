import os
from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI
import streamlit as st


os.environ["OPENAI_API_KEY"] = "e9217fc4-58e8-4fd4-acb8-b882ae5c2e71"
os.environ["OPENAI_API_BASE"] = "https://api.sambanova.ai/v1"
os.environ["OPENAI_MODEL_NAME"] = "sambanova/Meta-Llama-3.3-70B-Instruct"

custom_llm = ChatOpenAI(
    temperature=0.7,
    model=os.environ["OPENAI_MODEL_NAME"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

filter_agent = Agent(
    role="Prompt Filter",
    goal="Ensure user prompts are appropriate and doesnot contain any insecure words",
    backstory=(
        "You are a strict and ethical filter AI. You detect and reject prompts that contain any form of sexual content, "
        "violence, harassment, sensitive or offensive content, or pedophilia. Your mission is to allow only clean and "
        "safe prompts."
    ),
    verbose=True,
    allow_delegation=False,
    llm=custom_llm
)

def create_filter_task(user_prompt):
    return Task(
        description=(
            f"Review this prompt carefully: \"{user_prompt}\"\n\n"
            "Check if it contains **any** of the following types of content:\n"
            "- Sexual\n"
            "- Sensitive\n"
            "- Harassment or bullying\n"
            "- Violence or threats\n"
            "- Offensive language\n"
            "- Hate speech\n"
            "- Pedophilia\n\n"
            "- If it contains any of the listed inappropriate categories → Respond with 'The prompt is rejected'\n"
            "- Only if it's both clean and safe → Respond with 'Accepted prompt'"
            "The response must be either 'Accepted prompt' or 'The prompt is rejected'"
        ),
        expected_output="Either 'Accepted prompt' or 'The prompt is rejected'",
        agent=filter_agent
    )

def validate_prompt(user_prompt):
    task = create_filter_task(user_prompt)
    crew = Crew(
        agents=[filter_agent],
        tasks=[task],
        process=Process.sequential
    )
    result = crew.kickoff()
    print("\n🧾 Result:", result)
    return result

# Streamlit UI
st.set_page_config(page_title="Prompt Filter", layout="centered")
st.title("🛡️ Prompt Moderation System")

user_prompt = st.text_area("Enter your prompt for moderation:", height=150)

if st.button("Check Prompt"):
    if user_prompt.strip():
        with st.spinner("Validating..."):
            result = validate_prompt(user_prompt)
        st.success("Result:")
        st.markdown(f"**{result}**")
    else:
        st.warning("Please enter a prompt before clicking 'Check Prompt'.")


