# -*- coding:utf-8 -*-

# Web requests and parsers
import pickle
import requests
from bs4 import BeautifulSoup

# Importing sources
from components.sources import getTWI


def get_chapter_links(toc_link):
    try:
        page = requests.get(toc_link).text
        soup = BeautifulSoup(page, "html.parser")
        contents_tag = soup.find("div", "entry-content")
        chapter_links = [link.get("href") for link in contents_tag.find_all("a")]

        no_slash_links = []
        double_slash_links = []

        for link in chapter_links:
            if link[-1] != "/":
                no_slash_links.append(chapter_links.index(link))
            if link[-2:] == "//":
                double_slash_links.append(chapter_links.index(link))

        for indice in no_slash_links:
            chapter_links[indice] = chapter_links[indice] + "/"

        for indice in double_slash_links:
            chapter_links[indice] = chapter_links[indice][:-1]

        return chapter_links
    except:
        print(
            "You seem to be facing network issues, please check your internet connection, and retry..."
        )


def set_chapter_file_links(chapter_links):
    with open("assets\\toc", "wb") as f:
        pickle.dump(chapter_links, f)
        f.close()


if __name__ == "__main__":
    novel_details = getTWI()
    get_chapter_links(novel_details["TableOfContents"])
