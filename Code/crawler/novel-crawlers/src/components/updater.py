# -*- coding:utf-8 -*-
# Pickler
import pickle

# Chapter progress
from components.indexer import get_chapter_links, set_chapter_file_links

# Updating chapters based on current table of contents
def update_chapters():
    try:
        with open("assets\\toc", "rb") as f:
            links = pickle.load(f)
            f.close()
        link_list = get_chapter_links()
        if link_list != links:
            print("New chapters available")
            print("Updating chapters now...")
            set_chapter_file_links(link_list)
        else:
            print("Contents up to date")

    except FileNotFoundError as e:
        print("As file did not exist, we are currently creating the file...")
        set_chapter_file_links(get_chapter_links())
        print("File created. Run again.")
    except Exception as e:
        print(e)


# Main method to execute
if __name__ == "__main__":
    update_chapters()
