from typing import Final

# server-specific settings
HUGGING_FACE_DEVICE_MAP: Final[str] = "auto"

# program configurations
INITIAL_QUERY: Final[str] = "abs: Large Language Models"
LLM_MODEL_NAME_OR_MODEL_PATH: Final[str] = "path/to/your/model"
VECTOR_DB_SAVE_DIR: Final[str] = "vector_db"
CHAT_HISTORY_SAVE_DIR: Final[str] = "chat_history"
