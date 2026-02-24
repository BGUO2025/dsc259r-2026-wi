# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_book_links(text):
    # Initializing
    import bs4
    book_links = []

    # Point to HTML DOM tree
    bs_obj = bs4.BeautifulSoup(text, features='lxml')

    # Point to book list HTML structure
    book_list_ele = bs_obj.find_all(name='li', attrs={'class': 'col-xs-6 col-sm-4 col-md-3 col-lg-3'})

    # Iterate each book HTML and grab important info
    for book_block_ele in book_list_ele:
        # Grab info
        rating = (
            book_block_ele
            .find(name='p')
            .get('class')[1]
        )
        price = float(
            book_block_ele
            .find(name='p', attrs={'class': 'price_color'})
            .text
            .replace('£', '')
            .replace('Â', '')
        )
        
        # Conditionally store book link
        if (rating.lower() not in ['one', 'two', 'three']) and (price < 50):
            link = (
                book_block_ele
                .find(name='a')
                .get('href')
            )
            book_links.append(link) 

    return book_links  

def get_product_info(text, categories):
    # Initializing
    product_info_dict = {}
    category = ''

    # Point to HTML DOM tree
    bs_obj = bs4.BeautifulSoup(text, features='lxml')

    # Point to book store navigation menu structure
    navigation_menu_list = (
        bs_obj
        .find(name='ul', attrs={'class': 'breadcrumb'})
        .find_all(name='li')
    )
    # Grab the book category
    for i, list_item in enumerate(navigation_menu_list):
        if i == 2:
            category += (
                list_item
                .text
                .strip()
            )

    # Decide if this is the right book
    if category not in categories: return None

    # Grab other important info
    product_info_dict['Category'] = category

    product_info_dict['Title'] = (
        bs_obj
        .find(name='div', attrs={'class': 'col-sm-6 product_main'})
        .find(name='h1')
        .text
        .strip()
    )

    product_info_dict['Rating'] = (
        bs_obj
        .find(name='p', attrs={'class': 'instock availability'})
        .find_next_sibling('p')
        .get('class')[1]
        .strip()
    )

    product_info_dict['Description'] = (
        bs_obj
        .find(name='div', attrs={'id': 'product_description'})
        .find_next_sibling('p')
        .text
        .strip()
    )

    # You need to go into a deeper structure
    product_info_structure = (
        bs_obj
        .find(name='table', attrs={'class': 'table table-striped'})
        .find_all(name='tr')
    )
    for product_info_row in product_info_structure:
        # Grab the relevant indicator string
        product = (
            product_info_row
            .find(name='th')
            .text
            .strip()
        )
        # Need to check that's the right info we are looking for
        if product not in \
            ['UPC', 'Product Type', 'Price (excl. tax)', 'Price (incl. tax)', 'Tax', 'Availability', 'Number of reviews']: continue
        
        # Grab values
        product_value = (
            product_info_row
            .find(name='td')
            .text
            .strip()
        )      
        # Special treatment on price data
        if product in ['Price (excl. tax)', 'Price (incl. tax)', 'Tax']:
            product_value = (
                product_value
                .replace('£', '')
                .replace('Â', '')
            )
        product_info_dict[product] = product_value

    return product_info_dict
        
def scrape_books(k, categories):
    from collections import defaultdict
    # Initialize dictionary
    current_product_info = defaultdict(list)

    # Intialize URI parts
    protocol = 'http'
    path='books.toscrape.com/catalogue/'
    last_sub_path='page-'
    resource_type='html'

    # Access k websites for list of books
    for i in range(k):
        # Define URI
        book_list_uri = f'{protocol}://{path}{last_sub_path}{i+1}.{resource_type}'
        # Process HTTP response
        book_list_response_obj = requests.get(book_list_uri)
        book_list_HTML_content = book_list_response_obj.text
        # Parse the webpage and output for list of book links
        book_links = extract_book_links(book_list_HTML_content)
    
        # Access N websites for book links
        for link in book_links:
            # Define URI
            book_details_uri = f'{protocol}://{path}{link}'
            # Process HTTP response
            book_details_response_obj = requests.get(book_details_uri)
            book_details_HTML_content = book_details_response_obj.text
            # Parse the webpage and store product details info
            product_info = get_product_info(book_details_HTML_content, categories)
            
            # Combine product info across books
            if product_info is None: continue

            # Store info from each row
            for key in product_info:
                current_product_info[key].append(product_info[key])

    # Convert dict to Pandas df
    product_df = pd.DataFrame(current_product_info, columns=list(current_product_info.keys()))
    return product_df


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def get_comments(storyid):
    # Import this for easy appending for dict
    from collections import defaultdict

    # Initialize stack, visited stack, and result store
    stack = [(storyid, None)]
    visited = {storyid}
    comment_info = defaultdict(list)

    # DFS starts
    while stack:
        # LIFO
        (currentid, parent) = stack.pop()  
    
        # For each pop current node, get response from HTTP request    
        path = f'https://hacker-news.firebaseio.com/v0/item/{currentid}.json?print=pretty'
        response = requests.get(path).json()
        
        # Avoid the story info, just the comment info
        if (currentid != storyid) and not ('dead' in response and response['dead']):
            # Fetch relevant info
            comment_info['id'].append(currentid)
            comment_info['by'].append(response['by'])
            comment_info['text'].append(response.get('title', '') + response.get('text', ''))
            comment_info['parent'].append(parent)
            comment_info['time'].append(response['time'])

        # Find its neighboring node in the sense of graph structure
        for kid in response.get('kids', []):
            if kid in visited: continue
            stack.append((kid, currentid))
            visited.add(kid)

    # Convert from dict to Pandas df
    comment_info_df = pd.DataFrame(comment_info, columns=list(comment_info.keys()))
    comment_info_df['time'] = pd.to_datetime(comment_info_df['time'], unit='s')
    return comment_info_df