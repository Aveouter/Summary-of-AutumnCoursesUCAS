# -*- coding:utf-8 -*-

# Getting color into the picture
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Web requests and parsers
import pickle
import requests
from bs4 import BeautifulSoup

# Importing sources, and chapter links
from components.indexer import get_chapter_links
from components.sources import getTWI

# Counter function, which counts the words
def counter(index):
    totalWordCount = 0
    chapter_number = 0
    table = Table(title="The Wandering Inn Word Counter", caption=totalWordCount)

    table.add_column("Chapter", style="cyan")
    table.add_column("Title", style="magenta", no_wrap=True)
    table.add_column("Word Count", justify="right", style="green")

    print("[blue]A table will be printed for every 25 chapters.")
    print("[magenta]Thank you for using Novel Crawlers!")

    for link in track(index, description="[red]Counting"):
        page = requests.get(link).text
        soup = BeautifulSoup(page, "html.parser")
        chapter_number += 1
        chapter_title = soup.find("h1", "entry-title")
        chapter_title = chapter_title.get_text()
        chapter_tag = soup.find("article", "post")
        text = chapter_tag.findChildren("p")
        text = text[:-1]
        text = str(text)
        text = text[1:-1]
        text = text.replace("<p>", "")
        text = text.replace("</p>,", "")
        text = text.replace("</p>", "")
        text = str(text)
        words = text.split()

        table.add_row(
            f"Chapter {chapter_number:<5}",
            f"{chapter_title:<40}",
            f"{len(words)}"
        )

        totalWordCount += len(words)

        if chapter_number % 25 == 0:
            print(table)

    print("Total Word Count : ", totalWordCount)
    print("Number of pages : ", totalWordCount / 450)

    console = Console()
    console.print(table)


def worder():
    novel_details = getTWI()
    links = get_chapter_links(novel_details["TableOfContents"])
    counter(links)

if __name__ == "__main__":
    worder()
