import json
import os

import openai
import requests
import streamlit as st
from joblib import Parallel, delayed
from pydantic import BaseModel

###########################
# Define Pydantic Models  #
###########################


class PerplexityMessage(BaseModel):
    content: str


class PerplexityUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    citation_tokens: int
    num_search_queries: int


class PerplexityChoice(BaseModel):
    message: PerplexityMessage


class PerplexityResponse(BaseModel):
    citations: list[str]
    choices: list[PerplexityChoice]


###########################
# Global Config & Helpers #
###########################

# Get your Perplexity API key from the environment.
PERPLEXITY_API = os.getenv("PERPLEXITY_API")
if not PERPLEXITY_API:
    st.error("Missing PERPLEXITY_API environment variable.")
    st.stop()


# Define the function to call the Perplexity API.
def search_perplexity(prompt: str) -> PerplexityResponse:
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "search_domain_filter": [],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None,
    }
    headers = {"Authorization": f"Bearer {PERPLEXITY_API}", "Content-Type": "application/json"}
    response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers)
    return PerplexityResponse.model_validate_json(response.text)


########################################
# Define the Question Groups & Prompts #
########################################

questions1 = [
    "What are {company_name} core projects and recent developments?",
    "What are the main products or services that {company_name} offers?",
    "What recent projects or initiatives have {company_name} launched?",
    "Have {company_name} pivoted or expanded into new areas recently?",
    "What technologies or methodologies are {company_name} investing in?",
]
questions2 = [
    "How are {company_name} communicating their vision and direction?",
    "What kind of public speeches, interviews, or conference presentations have {company_name} leaders given?",
    "Have {company_name} published any white papers, research articles, or thought leadership pieces recently?",
    "How does {company_name} position themselves in industry discussions (e.g., sustainability, innovation, longevity)?",
]
questions3 = [
    "What funding rounds have {company_name} completed, and how much capital have they raised?",
    "Who are {company_name} investors, and what is the scale of their financial backing?",
    "Are there any publicly available financial reports or statements of {company_name}?",
]
questions4 = [
    "How reputable and influential is {company_name} in their industry?",
    "What awards, recognitions, or certifications has {company_name} received?",
    "How is {company_name} perceived by industry experts and analysts?",
    "What partnerships or collaborations do {company_name} have with other well-known organizations?",
    "Are there any customer testimonials, case studies, or independent reviews of {company_name}?",
]
questions5 = [
    "What are potential concerns or ethical considerations of {company_name}?",
    "Are there any controversies or ethical debates surrounding {company_name} products or services?",
    "What is the feedback from {company_name} customers and the broader public?",
    "Do {company_name} have clear policies regarding sustainability, ethical practices, or data privacy?",
]
questions6 = [
    "What is the {company_name} long-term vision and roadmap?",
    "How do {company_name} differentiate themselves from competitors?",
    "What are the key challenges {company_name} currently face?",
    "How do {company_name} handle innovation, and what is their approach to research and development?",
]

group_questions = {
    "What are their core projects and recent developments?": questions1,
    "How are they communicating their vision and direction?": questions2,
    "What is their financial health and funding status?": questions3,
    "How reputable and influential are they in their industry?": questions4,
    "What are potential concerns or ethical considerations?": questions5,
    "What additional aspects should I consider?": questions6,
}

########################################
# Streamlit Application Starts Here    #
########################################

st.title("Company Analysis and Conversation Starter Generator")

company_name = st.text_input("Enter the Company Name:", "")

if st.button("Analyze Company"):
    if not company_name.strip():
        st.error("Please enter a valid company name.")
    else:
        st.info(f"Analyzing company: **{company_name}**")
        company_info_all = {}  # To hold the results for all groups.
        overall_results = []  # For combining the responses later.

        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_groups = len(group_questions)
        current_group = 0

        # Process each question group
        for group_title, questions in group_questions.items():
            current_group += 1
            progress_text.markdown(f"Processing group {current_group} of {total_groups}: **{group_title}**")

            # Format the questions with the company name
            formatted_questions = [q.format(company_name=company_name) for q in questions]

            # Call the search_perplexity function in parallel
            try:
                responses = Parallel(n_jobs=-1)(
                    delayed(search_perplexity)(question) for question in formatted_questions
                )
            except Exception as e:
                st.error(f"Error retrieving data for group '{group_title}': {e}")
                continue

            # Build a dictionary of question->response content
            results = {}
            for resp, original_question in zip(responses, questions, strict=False):
                content = resp.choices[0].message.content
                # Replace citation placeholders with the actual citation text
                for i, citation in enumerate(resp.citations, start=1):
                    content = content.replace(f"[{i}]", f"({i}][{citation})")
                results[original_question] = content

            # Create a folder for this question group inside ~/Downloads/{company_name}/
            safe_group = group_title.strip().replace(" ", "_").replace("?", "")

            # Save the group results for combining later.
            company_info_all[group_title] = results
            overall_results.append(json.dumps(results, indent=4))

            progress_bar.progress(int(current_group / total_groups * 100))

        progress_text.text("All question groups processed.")
        st.success("Company data retrieval complete!")

        ########################################
        # Generate Conversation Topics         #
        ########################################

        st.info("Generating conversation topics for discussion...")
        # Combine all the retrieved info into a single string.
        combined_info = "\n--------------\n".join(overall_results)

        # Set your OpenAI API key (ensure OPENAI_API_KEY is set in your environment)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            st.error("Missing OPENAI_API_KEY environment variable.")
        else:
            try:
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="o3-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
I am a business development professional at a consulting company that specializes in Data, Machine Learning, and Software Engineering.
 Our company has a proven track record in AI and in developing scalable pipelines for processing and analyzing data.

I have an upcoming call with a new customer and I want to make sure I cover the following key areas:

1. **Identify New Business Opportunities**:  
   - What topics or questions can I use to uncover potential challenges or opportunities in their current processes?
   - How can I help them discover areas where our expertise in data, ML, and AI can add value?

2. **Develop and Nurture Client Relationships**:  
   - What conversation starters and questions will help build rapport and trust?
   - How can I express genuine interest in understanding their business, goals, and challenges?
   - What are some effective phrases to demonstrate our commitment to a long-term partnership?

3. **Drive the Sales Process**:  
   - How can I smoothly transition from understanding their needs to discussing how our solutions can meet those needs?
   - What key points should I emphasize to align our services with their business objectives?
   - What phrases or strategies will guide the conversation toward concrete next steps without being too aggressive?

Please generate a comprehensive discussion guide that includes:
- A list of topics and openeded questions for each of the three focus areas.
- Phrases and conversation starters tailored for a business development call.
- Strategies to ensure a natural flow from discovery to potential sales engagement.
- Any additional insights or best practices relevant to our industry.

Here is the additional information about the company: {combined_info}.


*Please format your response using Markdown for clarity and ease of reading.*
""",
                        },
                    ],
                    response_format={"type": "text"},
                    reasoning_effort="medium",
                )
                conversation_topics = response.choices[0].message.content

                st.markdown("## Conversation Starters and Topics")
                st.markdown(conversation_topics)

                # Create the content in memory
                file_content = f"{combined_info}\n--------------\n{conversation_topics}"

                # Create a download button with the content
                st.download_button(
                    label="Download Conversation Topics",
                    data=file_content,
                    file_name=f"{company_name}_questions_topics.txt",
                    mime="text/plain",
                )
                st.success("Click the button above to download the conversation topics")
            except Exception as e:
                st.error(f"Error generating conversation topics: {e}")
