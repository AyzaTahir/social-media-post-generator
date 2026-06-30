"""
Social Media Post Generator — Streamlit + CrewAI
Generate platform-specific social media posts from a topic, article, or rough idea.
"""

from __future__ import annotations

import os

# CrewAI injects cache_breakpoint for Anthropic caching; Groq rejects it.
# https://github.com/crewAIInc/crewAI/issues/5886
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import litellm

_UNSUPPORTED_LLM_MSG_KEYS = frozenset(
    {"cache_breakpoint", "is_litellm", "provider_specific_fields"}
)
_original_litellm_completion = litellm.completion


def _strip_unsupported_message_keys(kwargs: dict) -> dict:
    messages = kwargs.get("messages")
    if not messages:
        return kwargs
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            cleaned.append(
                {k: v for k, v in msg.items() if k not in _UNSUPPORTED_LLM_MSG_KEYS}
            )
        else:
            cleaned.append(msg)
    return {**kwargs, "messages": cleaned}


def _patched_litellm_completion(*args, **kwargs):
    kwargs = _strip_unsupported_message_keys(kwargs)
    return _original_litellm_completion(*args, **kwargs)


litellm.completion = _patched_litellm_completion

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Post Generator",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "linkedin_post": "",
    "twitter_post": "",
    "instagram_post": "",
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

PLATFORM_SPECS = {
    "LinkedIn": {
        "limit": "1300 characters",
        "style": (
            "professional but personable, can use short paragraphs and line breaks, "
            "1-3 relevant hashtags at the end, can include a hook line and a CTA "
            "like asking a question or inviting comments"
        ),
    },
    "Twitter/X": {
        "limit": "280 characters",
        "style": (
            "punchy, concise, conversational, no more than 1-2 hashtags, "
            "can use 1 relevant emoji if it fits naturally, hook in the first sentence"
        ),
    },
    "Instagram": {
        "limit": "2200 characters but keep it skimmable",
        "style": (
            "warm and engaging caption, can use emojis naturally, line breaks for "
            "readability, ends with 5-10 relevant hashtags on a new line, "
            "include a call-to-action (e.g. 'double tap', 'save this', 'comment below')"
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_llm(api_key: str, model: str) -> LLM:
    if not api_key or not api_key.strip():
        raise ValueError("API key is required. Paste your Groq API key in the sidebar.")
    return LLM(model=model, api_key=api_key.strip())


def run_social_post_crew(
    llm: LLM,
    *,
    topic: str,
    context: str,
    platforms: list[str],
    tone: str,
    goal: str,
) -> dict[str, str]:
    writer = Agent(
        role="Social Media Copywriter",
        goal=(
            "Write scroll-stopping, platform-native social media posts that match "
            "each platform's format and audience expectations."
        ),
        backstory=(
            "You are a senior social media strategist who has run content for "
            "B2B and B2C brands. You know that a LinkedIn post, a tweet, and an "
            "Instagram caption are NOT the same thing reformatted — each needs its "
            "own structure, pacing, and hook."
        ),
        llm=llm,
        verbose=False,
    )

    tasks = []
    for platform in platforms:
        spec = PLATFORM_SPECS[platform]
        tasks.append(
            Task(
                description=f"""
Write a {platform} post about the following:

**Topic / idea:** {topic}
**Extra context / article / notes:** {context or "None provided"}
**Tone:** {tone}
**Goal of this post:** {goal}

Platform requirements for {platform}:
- Length guideline: {spec['limit']}
- Style: {spec['style']}

Output ONLY the final post text, ready to copy-paste. Do not include any
preamble, labels, or explanation — just the post itself.
""",
                expected_output=f"The final {platform} post text, nothing else.",
                agent=writer,
            )
        )

    crew = Crew(
        agents=[writer],
        tasks=tasks,
        process=Process.sequential,
        llm=llm,
        verbose=False,
        tracing=False,
    )
    crew.kickoff()

    results = {}
    for platform, task in zip(platforms, tasks):
        output = task.output
        results[platform] = str(output.raw if hasattr(output, "raw") else output).strip()
    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Dashboard")
    st.caption("Configure your API key and post settings.")

    st.subheader("🔑 API Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        key="groq_api_key_input",
    )
    model_choice = st.selectbox(
        "Model",
        [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral-8x7b-32768",
        ],
        index=0,
    )

    st.divider()
    st.subheader("📱 Platforms")
    selected_platforms = st.multiselect(
        "Generate posts for",
        list(PLATFORM_SPECS.keys()),
        default=list(PLATFORM_SPECS.keys()),
    )

    st.divider()
    st.subheader("🎯 Post Settings")
    tone = st.selectbox(
        "Tone",
        ["Professional", "Friendly", "Bold/confident", "Witty/humorous", "Inspirational"],
    )
    goal = st.selectbox(
        "Goal",
        [
            "Drive engagement (likes/comments)",
            "Drive clicks/traffic to a link",
            "Build personal brand / thought leadership",
            "Announce something (launch, milestone, event)",
            "Educate / share a tip",
        ],
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📱 Social Media Post Generator")
st.markdown(
    "Turn one idea into **platform-native posts** for LinkedIn, Twitter/X, and "
    "Instagram — each written in that platform's own voice, not just copy-pasted."
)

topic = st.text_area(
    "What's the post about?",
    placeholder="e.g. We just hit 1,000 users for our app, or: 5 lessons from my first year freelancing",
    height=80,
)
context = st.text_area(
    "Extra context (optional)",
    placeholder="Paste an article, blog post, notes, stats, or any extra details to ground the post in.",
    height=120,
)

generate_btn = st.button("🚀 Generate Posts", type="primary", use_container_width=True)

if generate_btn:
    if not topic.strip():
        st.error("Tell me what the post is about first.")
    elif not selected_platforms:
        st.error("Select at least one platform in the sidebar.")
    else:
        with st.spinner("CrewAI is writing your posts..."):
            try:
                llm = build_llm(groq_api_key, model_choice)
                results = run_social_post_crew(
                    llm,
                    topic=topic,
                    context=context,
                    platforms=selected_platforms,
                    tone=tone,
                    goal=goal,
                )
                for platform, text in results.items():
                    key = f"{platform.lower().split('/')[0]}_post"
                    st.session_state[key] = text
            except Exception as exc:
                st.error(str(exc))

st.divider()

tab_map = {
    "LinkedIn": "linkedin_post",
    "Twitter/X": "twitter_post",
    "Instagram": "instagram_post",
}
visible_tabs = [p for p in PLATFORM_SPECS if st.session_state.get(tab_map[p])]

if visible_tabs:
    tabs = st.tabs([f"{'💼' if p=='LinkedIn' else '🐦' if p=='Twitter/X' else '📸'} {p}" for p in visible_tabs])
    for tab, platform in zip(tabs, visible_tabs):
        with tab:
            text = st.session_state[tab_map[platform]]
            st.text_area(f"{platform} post", value=text, height=260, key=f"display_{platform}")
            st.caption(f"{len(text)} characters")
else:
    st.info("Fill in the topic above and click **Generate Posts** to see results here.")
