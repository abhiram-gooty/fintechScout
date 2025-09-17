import os
import json
import re
import requests
from typing import TypedDict, List, Dict, Any
import streamlit as st
import pandas as pd
from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOllama
import time

# ==========================
# State Definition
# ==========================
class ScoutState(TypedDict):
    queries: List[str]
    articles: List[Dict[str, Any]]
    companies: List[str]
    insights: List[str]
    scores: Dict[str, Any]
    report: str
    company_sources: Dict[str, str]

# ==========================
# LLM Setup
# ==========================
try:
    llm = ChatOllama(model="mistral", temperature=0.2)
    st.write("✅ Ollama LLM initialized successfully")
except Exception as e:
    st.warning(f"⚠️ Ollama not available: {e}")
    st.info("Using mock responses for demonstration")

    class MockLLM:
        def invoke(self, prompt):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            st.write("Mock LLM invoked for prompt:", prompt[:100], "...")
            return MockResponse(json.dumps({
                "companies": ["HTX Pay", "RBI"],
                "insights": [
                    "HTX Pay is innovating in crypto payments.",
                    "RBI is introducing new regulatory guidelines."
                ],
                "scores": {
                    "HTX Pay": {"score": 8, "reason": "Strong fintech growth potential"},
                    "RBI": {"score": 7, "reason": "Regulatory authority"}
                }
            }))
    llm = MockLLM()

# ==========================
# News Fetcher
# ==========================
def search_news(queries: List[str]) -> List[Dict[str, str]]:
    api_key = os.environ.get("GNEWS_API_KEY", "93198eda0a634acd2ef1559184f9ec93")
    articles = []
    for q in queries:
        url = f"https://gnews.io/api/v4/search?q={q}&lang=en&max=5&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("articles", []):
                    articles.append({
                        "title": item["title"],
                        "url": item["url"],
                        "description": item.get("description", "")
                    })
        except Exception as e:
            st.error(f"Error fetching news: {e}")
    if not articles:
        articles = [
            {
                "title": "Stripe Raises $6.5B in Latest Funding Round",
                "url": "https://example.com/stripe-funding",
                "description": "Payment processor Stripe has secured $6.5 billion in new funding."
            }
        ]
    st.write(f"📰 Articles fetched: {len(articles)}")
    return articles

# ==========================
# Helpers
# ==========================
def safe_json_extract(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return {}

fintech_keywords = ['payment', 'fintech', 'crypto', 'UPI', 'bank', 'RBI', 'digital', 'transaction']

def is_fintech(text):
    text = text.lower()
    return any(keyword in text for keyword in fintech_keywords)

# ==========================
# Agents
# ==========================
def news_scout(state: ScoutState) -> ScoutState:
    start = time.time()
    st.write("⏳ Running news_scout...")
    state["articles"] = search_news(state["queries"])
    st.write(f"✅ news_scout completed in {time.time() - start:.2f}s")
    return state

def research_analyst(state: ScoutState) -> ScoutState:
    start = time.time()
    st.write("⏳ Running research_analyst...")
    companies, insights, company_sources = [], [], {}
    for art in state["articles"]:
        prompt = f"""
        You are a financial research analyst. Extract company names and investment insights from this article.
        Respond ONLY in JSON format.

        Article Title: {art['title']}
        URL: {art['url']}
        Content: {art['description']}
        """
        resp = llm.invoke(prompt)
        parsed = safe_json_extract(resp.content)
        st.write("Parsed response:", parsed)

        if parsed.get("companies"):
            for comp in parsed["companies"]:
                companies.append(comp)
                company_sources[comp] = art["url"]
        else:
            found = re.findall(r"\b[A-Z][A-Za-z0-9&\-]+(?:\s[A-Z][A-Za-z0-9&\-]+)*\b", art["title"])
            for comp in found[:2]:
                companies.append(comp)
                company_sources[comp] = art["url"]

        insights.extend(parsed.get("insights", []))

    state["companies"] = list(set(companies))
    state["insights"] = insights
    state["company_sources"] = company_sources
    st.write(f"✅ research_analyst completed in {time.time() - start:.2f}s")
    return state

def investment_analyst(state: ScoutState) -> ScoutState:
    start = time.time()
    st.write("⏳ Running investment_analyst...")
    scores = {}
    for comp in state["companies"]:
        prompt = f"""
        You are an investment analyst. Rate the investment potential of {comp}
        on a scale of 1 to 10. Respond ONLY in JSON format.
        """
        resp = llm.invoke(prompt)
        parsed = safe_json_extract(resp.content)
        if not parsed:
            parsed = {"score": 5, "reason": "Fallback reason"}
        # Debugging
        st.write(f"Scoring {comp}: {parsed}")
        scores[comp] = parsed
    state["scores"] = scores
    st.write(f"✅ investment_analyst completed in {time.time() - start:.2f}s")
    return state

def report_writer(state: ScoutState) -> ScoutState:
    start = time.time()
    st.write("⏳ Running report_writer...")
    state["report"] = "Report generated."
    st.write(f"✅ report_writer completed in {time.time() - start:.2f}s")
    return state

# ==========================
# Streamlit UI
# ==========================
st.title("🔍 Fintech Scouting Dashboard")

query_input = st.text_input("Enter a fintech-related query:", "")
start_button = st.button("Start Scouting")

if start_button and query_input.strip():
    state = ScoutState(
        queries=[query_input],
        articles=[],
        companies=[],
        insights=[],
        scores={},
        report="",
        company_sources={}
    )
    graph = StateGraph(ScoutState)
    graph.add_node("news_scout", news_scout)
    graph.add_node("research_analyst", research_analyst)
    graph.add_node("investment_analyst", investment_analyst)
    graph.add_node("report_writer", report_writer)
    graph.set_entry_point("news_scout")
    graph.add_edge("news_scout", "research_analyst")
    graph.add_edge("research_analyst", "investment_analyst")
    graph.add_edge("investment_analyst", "report_writer")
    app = graph.compile()

    try:
        start_total = time.time()
        final_state = app.invoke(state)
        st.write(f"✅ Total pipeline completed in {time.time() - start_total:.2f}s")

        # Filter fintech content
        filtered_articles = [a for a in final_state["articles"] if is_fintech(a['title']) or is_fintech(a['description'])]
        filtered_companies = [c for c in final_state["companies"] if c in final_state["scores"]]
        filtered_insights = [i for i in final_state["insights"] if is_fintech(i)]

        st.subheader("✅ Recommended Companies")
        if filtered_companies:
            table_data = []
            for comp in filtered_companies:
                score_data = final_state["scores"].get(comp, {})
                reason = score_data.get('reason') or score_data.get('Rationale') or score_data.get('Explanation') or 'No reason provided'
                score = score_data.get('score') or score_data.get('Investment Potential') or score_data.get('Rating') or 'N/A'
                source = final_state["company_sources"].get(comp, '#')
                table_data.append({
                    "Company": comp,
                    "Reason": reason,
                    "Score": score,
                    "Source": source
                })
            df = pd.DataFrame(table_data)
            st.table(df)
        else:
            st.write("No companies found.")

        st.subheader("📄 Relevant Articles")
        if filtered_articles:
            for article in filtered_articles:
                st.markdown(f"### [{article['title']}]({article['url']})")
                st.write(article['description'])
        else:
            st.write("No articles found.")

        st.subheader("💡 Insights")
        if filtered_insights:
            for insight in filtered_insights:
                st.write(f"- {insight}")
        else:
            st.write("No insights found.")

        st.subheader("🌟 More Trending Fintech Topics")
        recommendations = ["Blockchain-based financing", "AI fraud detection"]
        for rec in recommendations:
            st.write(f"- {rec}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    if not query_input.strip():
        st.warning("Please enter a query to begin.")
