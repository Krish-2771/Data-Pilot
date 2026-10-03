"""Registry of the Page objects used by the app's st.navigation router."""

from typing import Any

_pages_by_title: dict[str, Any] = {}


def register_pages(pages: list[Any]) -> None:
    """Store the exact Page objects passed to ``st.navigation``."""
    global _pages_by_title
    _pages_by_title = {page.title: page for page in pages}


def get_page(title: str) -> Any:
    """Return a registered Page by its visible title."""
    try:
        return _pages_by_title[title]
    except KeyError as exc:
        raise RuntimeError(f"Streamlit page is not registered: {title}") from exc
