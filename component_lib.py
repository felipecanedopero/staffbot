import streamlit.components.v1 as components
from pathlib import Path

parent_dir = Path(__file__).parent.absolute() / "st_autoprompt" / "autoprompt_component"
build_dir = parent_dir / "frontend/build"

auto_prompt = components.declare_component("auto_prompt", path=str(build_dir))
