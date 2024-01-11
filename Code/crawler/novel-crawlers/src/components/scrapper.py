# -*- coding:utf-8 -*-
# Accessing components
import components.extractor as extractor

# Zipfile creation modules
import zipfile

# Sources and Extractors
from components.indexer import get_chapter_links
from components.sources import getTWI

# Directory traversal
import os


# Builds the EPUB File
class ContentBuilder:
    # Defines output folder and book name
    def __init__(self, ChapterName, NovelName, Author, TableOfContents, OutputFolder):
        self.ChapterName = ChapterName
        self.NovelName = NovelName
        self.Author = Author
        self.TableOfContents = TableOfContents

        self.OutputFolder = OutputFolder

    # Generates the xhtml files
    def web(self):
        output_folder = self.OutputFolder

        link_list = get_chapter_links(self.TableOfContents)

        file_list = []
        for x in range(len(link_list)):
            namer = link_list[x][36:-1]
            print(namer)
            extractor.download(
                link_list[x], os.path.join(output_folder, str(x) + ".html")
            )
            extractor.clean(
                os.path.join(output_folder, str(x) + ".html"),
                os.path.join(
                    output_folder, self.ChapterName + str(namer) + ".xhtml"
                ),
            )
            file_list.append(
                os.path.join(output_folder, self.ChapterName + str(namer) + ".xhtml")
            )
        extractor.generate(file_list, self.OutputFolder, self.NovelName, self.Author)


if __name__ == "__main__":
    novel_details = getTWI()
    NovelBuilder = ContentBuilder(
        novel_details["ChapterName"],
        novel_details["NovelName"],
        novel_details["Author"],
        novel_details["TableOfContents"],
    )
