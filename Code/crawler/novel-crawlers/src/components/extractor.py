# -*- coding:utf-8 -*-

# Requesting and Parsing Webpages Imports
import requests
from bs4 import BeautifulSoup

# Directory Traversal
import os

# Zipfile creation
import zipfile


# Downloading the webpage content
def download(link, file_name):
    page = requests.get(link).text
    file = open(file_name, "w", encoding="utf-8")
    file.write(page)
    file.close()


# Parsing the webpage content
def clean(file_name_in, file_name_out):
    # Parsing and Cleaning Content
    raw = open(file_name_in, "r", encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    raw.close()
    chapter_tag = soup.find("article", "post")
    chapter_title = soup.find("h1", "entry-title")
    chapter_title = chapter_title.get_text()
    text = chapter_tag.findChildren("p")
    text = text[:-1]
    text = str(text)
    text = text[1:-1]
    text = text.replace("</p>,", "</p>\n")

    # Writing to XML file format
    f = open(file_name_out, "w", encoding="utf-8")
    f.write(
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" epub:prefix="z3998: https://daisy.org/z3998/2012/vocab/structure/" lang="en" xml:lang="en">'
    )
    f.write("\n<head>")
    f.write("\n<title>" + chapter_title + "</title>")
    f.write("\n</head>")
    f.write('\n<body dir="default">')
    f.write("\n<h1>" + chapter_title + "</h1>")
    f.write(text)
    f.write("\n</body>")
    f.write("\n</html>")
    f.close()
    os.remove(file_name_in)


def find_between(file):
    f = open(file, "r", encoding="utf8")
    soup = BeautifulSoup(f, "html.parser")
    return soup.title


# Creating the EPUB file
def generate(html_files, output_folder, novelname, author):
    epub = zipfile.ZipFile(output_folder + novelname + ".epub", "w")
    epub.writestr("mimetype", "application/epub+zip")
    epub.writestr(
        "META-INF/container.xml",
        """<container version="1.0"
    xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
        <rootfiles>
            <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
        </rootfiles>
    </container>""",
    )

    index_tpl = """<package version="3.1" xmlns="http://www.idpf.org/2007/opf">
    <metadata>
        %(metadata)s
    </metadata>
    <manifest>
        %(manifest)s
    </manifest>
    <spine>
        <itemref idref="toc" linear="no"/>
        %(spine)s
    </spine>
</package>"""
    manifest = ""
    spine = ""

    metadata = """<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">%(novelname)s</dc:title>\n<dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:ns0="http://www.idpf.org/2007/opf" ns0:role="aut" ns0:file-as="NaN">%(author)s</dc:creator>\n<meta xmlns:dc="http://purl.org/dc/elements/1.1/" name="calibre:series" content="%(series)s"/>""" % {
        "novelname": novelname,
        "author": author,
        "series": novelname,
    }

    toc_manifest = '<item href="toc.xhtml" id="toc" properties="nav" media-type="application/xhtml+xml"/>'

    for i, html in enumerate(html_files):
        basename = os.path.basename(html)
        manifest += (
            '<item id="file_%s" href="%s" media-type="application/xhtml+xml"/>\n'
            % (i + 1, basename)
        )
        spine += '<itemref idref="file_%s" />' % (i + 1)
        epub.write(html, "OEBPS/" + basename)

    epub.writestr(
        "OEBPS/content.opf",
        index_tpl
        % {
            "metadata": metadata,
            "manifest": manifest + toc_manifest,
            "spine": spine,
        },
    )

    toc_start = """<?xml version='1.0' encoding='utf-8'?>
    <!DOCTYPE html>
    <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
        <head>
            <title>%(novelname)s</title>
        </head>
            <body>
                <section class="frontmatter TableOfContents">
            <header>
                <h1>Contents</h1>
            </header>
                <nav id="toc" role="doc-toc" epub:type="toc">
                    <ol>
                        %(toc_mid)s
                        %(toc_end)s"""
    toc_mid = ""
    toc_end = """</ol></nav></section></body></html>"""

    for i, y in enumerate(html_files):
        chapter = find_between(html_files[i])
        chapter = str(chapter)
        toc_mid += """<li class="toc-Chapter-rw" id="num_%s">
                    <a href="%s">%s</a>
                    </li>""" % (
            i,
            os.path.basename(y),
            chapter,
        )

    epub.writestr(
        "OEBPS/toc.xhtml",
        toc_start % {"novelname": novelname, "toc_mid": toc_mid, "toc_end": toc_end},
    )
    epub.close()

    # commented code below as the files do not need to be deleted, in dev mode.
    for x in html_files:
        os.remove(x)
