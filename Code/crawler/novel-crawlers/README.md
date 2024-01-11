# Novel Crawlers
Web crawlers built to get your light novel content from the web, to you.

# *THIS PROJECT IS NOW BEING ARCHIVED*

Initially, this work was only made as a means for a young programmer to learn to make an EPUB for the book that he loves. Now, I've passed that stage and the book too has continued to grow larger and the author more skilled. The website that I parse for EPUB generation has evolved and despite me being able to work around it, I see no point to creating the EPUBs when the internet is largely available, and my goal for spreading propaganda has been achieved.

Please consider supporting the book's author by reading from the website instead. Recently, the content has only gotten much more interesting with a bunch of effects that I suppose are only possible on the website, and I wish for anyone reading to be able to experience that, at it's fullest.

This repository and novel-crawlers v2, however, will stay on GitHub as the remains of my first passion project.

### Features

- [x] Crawling the page and returning content
- [ ] Offering light novel options available in the site
- [ ] Downloading the light novels in different formats.

### Books
- [x] [The Wandering Inn](https://wanderinginn.com)
- [ ] [Worm](https://parahumans.wordpress.com)

### A little about novel-crawlers
The purpose for making this repository was for the convenience of being able to read the books on my phone offline, as I'm used to travelling a lot and mobile data is quite the issue.
Having found no epubs online for download, I've sought out to make this repository with a few python scripts to let me get my content.

I hold no power over the content, and all credit to the books and the amazing content inside them, should be given to the authors.\
So, please support them, as this is simply a project made for convenience and nothing else.

## Usage
The flow of utilization of novel crawlers is through the `scrapper.py`, which loads in all the components from the different python files and executes them generating us an EPUB file to use. \
The `scrapper.py` creates a class for the Book and the web() function takes in the table of contents as the input and downloads the content in the form of `.xhtml` files. \
These `.xhtml` files are then placed into a zip file along with a few other files creating an EPUB file.
These folders and files are,
 - mimetype => Used to identify what kind of file the reader is dealing with, application/epub+zip in this case.
 - container.xml => Creates a structure in the form of a container for the EPUB File.
 - Content.opf => Metadata.
 - toc.ncx => Table of Contents.

We have also created a Dockerfile, which can be used as a means to get the server set up and running. \
In order to build an image from source, the command is,
```sh
docker build -t novel-crawler .
```

Slight modifications will have to be made on your part for where you want the book to be downloaded to, which is the `path/to/files`. \
**docker-compose** \
```Dockerfile
version: '2.2'
services:
  novel-crawlers:
    container_name: novel-crawler
    image: datadi/novel-crawlers:latest
    volumes:
      - <path/to/files>:/data/files
    ports:
      - "10000:10000"
    restart: unless-stopped
```

Next, we can use a `docker-compose up` to run the container. \
The documentation for the various routes are present at `http://0.0.0.0:10000/docs` and `http://0.0.0.0:10000/redoc`. \
But, I'll save you some time and tell you which is the request you want to be executing.
```sh
curl -X 'POST' \
  'http://0.0.0.0:10000/scrape?output_path=%2Fdata%2Ffiles%2F' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "BookName": "wandering-inn",
  "ChapterName": "twi-",
  "NovelName": "The Wandering Inn",
  "Author": "pirateaba",
  "TableOfContents": "https://wanderinginn.com/table-of-contents/"
}'
```

## License
This repository is hosted under the MIT License, and is available for use.
