def getTWI():
    return {
        "ChapterName": "twi-",
        "NovelName": "The Wandering Inn",
        "Author": "pirateaba",
        "TableOfContents": "https://wanderinginn.com/table-of-contents/",
    }


def getWorm():
    return {
        "ChapterName": "worm-",
        "NovelName": "Worm",
        "Author": "WildBowPig",
        "TableOfContents": "https://parahumans.wordpress.com/table-of-contents/",
    }


def getNovelDetails(requestedNovel):
    library = {1: getTWI(), 2: getWorm()}
    return library.get(requestedNovel)


if __name__ == "__main__":
    book = getNovelDetails(2)
    print(book)
