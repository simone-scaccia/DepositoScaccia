"""Utilities for slugifying text and performing a DuckDuckGo search.

This module exposes:
- ``slugify``: convert text to a URL-friendly slug.
- ``search_ddgs``: perform a text search via ddgs and print results.
"""

import sys
from ddgs import DDGS


# Write a slugify function from text
def slugify(text: str) -> str:
    """
    Convert a string to a URL-friendly slug:
    - Lowercase using Unicode-aware casefold
    - Replace ASCII space characters with hyphens

    Parameters
    ----------
    text : str
        The input string to be slugified.

    Returns
    -------
    str
        The slugified string, with spaces replaced by hyphens and all characters in lowercase.

    Examples
    --------
    >>> slugify("Hello World")
    'hello-world'
    """
    # Fast return for empty strings
    if not text:
        return text

    # Unicode-aware lowercasing
    lowered = text.casefold()

    # Avoid replace() when there are no ASCII spaces
    if " " not in lowered:
        return lowered

    # Replace ASCII space characters with hyphens
    return lowered.replace(" ", "-")



def search_ddgs(query: str, max_results: int = 3):
    """
    Search DuckDuckGo using the ddgs client and print formatted results.

    Parameters
    ----------
    query : str
        The search query string.
    max_results : int, optional
        Maximum number of results to return (default is 5, min 1, max 50).

    Returns
    -------
    list[dict[str, str]]
        A list of dicts containing results with keys like "title", "href",
        and "body".

    Raises
    ------
    RuntimeError
        If the duckduckgo_search package is not installed.

    Examples
    --------
    >>> results = search_ddgs(max_results=3)
    >>> for r in results:
    ...     print(r["title"], r["href"])
    """

    if not query or not query.strip():
        if __name__ == "__main__":
            print("No query provided.")
            sys.exit(0)
        return []

    constrained_max = max(1, min(int(max_results), 50))

    if DDGS is None:  # pragma: no cover
        raise RuntimeError(
            "ddgs is required. Install with 'pip install duckduckgo_search'."
        )

    with DDGS(verify=False) as ddgs:
        results_iter = ddgs.text(query, max_results=constrained_max)
        results = list(results_iter)

    if not results:
        print("No results found.")
        sys.exit(0)

    for idx, item in enumerate(results, start=1):
        title = item.get("title") or item.get("name") or "(no title)"
        url = item.get("href") or item.get("url") or ""
        snippet = item.get("body") or item.get("snippet") or ""
        print(f"{idx}. {title}\n   {url}")
        if snippet:
            print(f"   {snippet}")

    return results

def main() -> None:
    """CLI entry point for interactive search input."""
    try:
        user_query = input("Enter search query: ").strip()
    except EOFError:
        user_query = ""

    if not user_query:
        print("No query provided.")
        sys.exit(0)

    search_ddgs(user_query)


if __name__ == "__main__":
    main()
