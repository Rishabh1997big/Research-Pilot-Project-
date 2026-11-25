import os

# Tests must not send traces to LangSmith. Set before app.config loads .env (load_dotenv never overrides
# variables that are already set).
os.environ["LANGSMITH_TRACING"] = "false"
