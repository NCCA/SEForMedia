"""
Export marimo notebooks to markdown and push them into Joplin.

This is run as a pre-commit hook (see .pre-commit-config.yaml). pre-commit
passes in the staged .py files, I pick out the ones that are marimo
notebooks, run ``marimo export md`` on each into MarkdownExports/ (same
folder layout as the repo) and then, if the Joplin desktop app is running
with the Web Clipper service turned on, create or update a note for each one
under the SEForMedia notebook, with one sub-notebook per repo folder.

Run it by hand with ``--all`` to export every notebook in the repo.

The Joplin side uses the Data API
(https://joplinapp.org/help/api/references/rest_api/). The token is read from
the JOPLIN_TOKEN environment variable, or failing that from the desktop app's
settings.json. If Joplin isn't there the export still happens and the hook
still passes, as I don't want a closed app to block a commit.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "MarkdownExports"
JOPLIN_NOTEBOOK = "SEForMedia"
JOPLIN_URL = os.environ.get("JOPLIN_URL", "http://localhost:41184")
JOPLIN_SETTINGS = Path.home() / ".config" / "joplin-desktop" / "settings.json"

# marimo writes the notebook header as YAML front matter, which Joplin shows
# as raw text, so it is stripped before the note is sent.
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)


def is_marimo_notebook(path: Path) -> bool:
    try:
        return "marimo.App(" in path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def all_notebooks() -> list[Path]:
    files = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO_ROOT / f for f in files]


def export_notebook(notebook: Path) -> Path:
    """
    Export a single notebook to markdown.

    Parameters
    ----------
        notebook : Path
            the marimo .py file to export

    Returns
    -------
        Path
            the markdown file written into OUTPUT_DIR
    """
    relative = notebook.resolve().relative_to(REPO_ROOT)
    output = OUTPUT_DIR / relative.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "md",
            str(notebook),
            "-o",
            str(output),
            "-f",
        ],
        check=True,
        capture_output=True,
    )
    return output


class Joplin:
    """
    Minimal wrapper around the Joplin Data API.

    Attributes
    ----------
        token : str
            API token from the Web Clipper settings
        folders : list[dict]
            cached list of every notebook, fetched once
    """

    def __init__(self, token: str) -> None:
        self.token = token
        self.folders = self._get_all("folders", fields="id,title,parent_id")

    def _request(
        self, method: str, path: str, data: dict | None = None, **query: str
    ) -> dict:
        query["token"] = self.token
        url = f"{JOPLIN_URL}/{path}?{urllib.parse.urlencode(query)}"
        body = json.dumps(data).encode() if data is not None else None
        request = urllib.request.Request(url, data=body, method=method)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read() or b"{}")

    def _get_all(self, path: str, **query: str) -> list[dict]:
        # the API pages results, so keep asking until has_more is false
        items, page = [], 1
        while True:
            result = self._request("GET", path, page=str(page), **query)
            items += result["items"]
            if not result.get("has_more"):
                return items
            page += 1

    def folder(self, title: str, parent_id: str = "") -> str:
        """
        Find a notebook by title under parent_id, creating it if it isn't there.

        Parameters
        ----------
            title : str
                notebook name
            parent_id : str
                id of the parent notebook, empty for a top level one

        Returns
        -------
            str
                the notebook id
        """
        for f in self.folders:
            if f["title"] == title and f["parent_id"] == parent_id:
                return f["id"]
        created = self._request(
            "POST", "folders", {"title": title, "parent_id": parent_id}
        )
        self.folders.append(
            {"id": created["id"], "title": title, "parent_id": parent_id}
        )
        return created["id"]

    def upsert_note(self, folder_id: str, title: str, body: str) -> str:
        notes = self._get_all(f"folders/{folder_id}/notes", fields="id,title")
        for note in notes:
            if note["title"] == title:
                self._request("PUT", f"notes/{note['id']}", {"body": body})
                return "updated"
        self._request(
            "POST", "notes", {"title": title, "parent_id": folder_id, "body": body}
        )
        return "created"


def joplin_token() -> str | None:
    if token := os.environ.get("JOPLIN_TOKEN"):
        return token
    try:
        return json.loads(JOPLIN_SETTINGS.read_text()).get("api.token")
    except (OSError, json.JSONDecodeError):
        return None


def connect_to_joplin() -> Joplin | None:
    token = joplin_token()
    if not token:
        print("Joplin: no API token found, skipping import")
        return None
    try:
        with urllib.request.urlopen(f"{JOPLIN_URL}/ping", timeout=2) as response:
            if response.read() != b"JoplinClipperServer":
                raise urllib.error.URLError("unexpected ping reply")
        return Joplin(token)
    except (urllib.error.URLError, OSError) as error:
        print(f"Joplin: not reachable at {JOPLIN_URL} ({error}), skipping import")
        return None


def push_to_joplin(joplin: Joplin, notebook: Path, markdown: Path) -> None:
    relative = notebook.resolve().relative_to(REPO_ROOT)
    parent_id = joplin.folder(JOPLIN_NOTEBOOK)
    # one sub-notebook per repo folder, nested for things like MNIST/Sketch
    for part in relative.parent.parts:
        parent_id = joplin.folder(part, parent_id)
    body = FRONT_MATTER.sub("", markdown.read_text(encoding="utf-8"))
    body += f"\n\n---\nExported from `{relative.as_posix()}`\n"
    status = joplin.upsert_note(parent_id, relative.stem, body)
    print(f"Joplin: {status} {JOPLIN_NOTEBOOK}/{relative.with_suffix('').as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "files", nargs="*", type=Path, help="files to check (from pre-commit)"
    )
    parser.add_argument(
        "--all", action="store_true", help="export every marimo notebook in the repo"
    )
    parser.add_argument(
        "--no-joplin", action="store_true", help="only write the markdown files"
    )
    args = parser.parse_args()

    candidates = all_notebooks() if args.all else args.files
    notebooks = [f for f in candidates if f.suffix == ".py" and is_marimo_notebook(f)]
    if not notebooks:
        return 0

    joplin = None if args.no_joplin else connect_to_joplin()
    failed = False
    for notebook in notebooks:
        try:
            markdown = export_notebook(notebook)
            print(f"exported {markdown.relative_to(REPO_ROOT)}")
        except subprocess.CalledProcessError as error:
            print(f"failed to export {notebook}: {error.stderr.decode().strip()}")
            failed = True
            continue
        if joplin is not None:
            try:
                push_to_joplin(joplin, notebook, markdown)
            except (urllib.error.URLError, OSError, KeyError) as error:
                print(f"Joplin: failed on {notebook} ({error})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
