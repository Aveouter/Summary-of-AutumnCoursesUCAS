# Importing enum for setting up options
from enum import Enum

# Importing FastAPI Tools
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Importing the scraper
from components.scrapper import ContentBuilder

# Initializing a FastAPI application
app = FastAPI()

# Defining a Book Title model
class BookTitle(str, Enum):
    twi = "wandering-inn"
    worm = "worm"

# Defining the Novel model
class Novel(BaseModel):
    BookName: str
    ChapterName: str
    NovelName: str
    Author: str
    TableOfContents: str

@app.get('/')
async def home_page() -> [BookTitle]:
    """Providing Books to choose from"""
    book_titles = [BookTitle.twi, BookTitle.worm]
    return book_titles

@app.post('/add-book')
async def add_book(novel: Novel) -> Novel:
    return novel

@app.get('/books/{book_name}')
async def get_book(book_name: str):
    if book_name == "wandering-inn":
        result = {
            "BookName": "wandering-inn",
            "ChapterName": "twi-",
            "NovelName": "The Wandering Inn",
            "Author": "pirateaba",
            "TableOfContents": "https://wanderinginn.com/table-of-contents/",
        }
    elif book_name == "worm":
        result =  {
            "BookName": "worm",
            "ChapterName": "worm-",
            "NovelName": "Worm",
            "Author": "WildBowPig",
            "TableOfContents": "https://parahumans.wordpress.com/table-of-contents/",
        }
        return RedirectResponse("/not-found")
    else:
        result = {"message": "help"}

    return result

@app.post('/scrape')
async def scrape_book(novel: Novel, output_path: str):
    ContentBuilder(
        novel.ChapterName,
        novel.NovelName,
        novel.Author,
        novel.TableOfContents,
        output_path
    ).web()

@app.get('/not-found')
async def not_found():
    # This page should be used for options that aren't made yet.
    return {
            "message": "The world works in mysterious ways.",
            "message2": "One day, you're the guy who has everything figured out.",
            "message3": "The next day, even your dog is having an existential crisis."
            }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)
